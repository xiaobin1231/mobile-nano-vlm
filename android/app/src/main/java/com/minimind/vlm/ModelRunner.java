package com.minimind.vlm;

import android.content.Context;
import android.content.Intent;
import android.net.LocalSocket;
import android.net.LocalSocketAddress;
import android.os.Handler;
import android.os.Looper;
import androidx.core.content.ContextCompat;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/** Serial, streaming client for the resident minimind_cli server. */
public final class ModelRunner implements AutoCloseable {
    private static final String INTERNAL_SOCKET_NAME = "minimind_vlm";
    private static final int MAX_FRAME_SIZE = 1024 * 1024;
    private static final long CONNECT_TIMEOUT_MS = 30_000;

    private final Context appContext;
    private final ModelStorage storage;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    public interface Callback {
        void onOutput(String text);
        void onComplete(InferenceStats stats);
        void onError(String error);
    }

    public static final class InferenceStats {
        public final long totalMs;
        public final double imageLoadMs;
        public final double preprocessMs;
        public final double visionMs;
        public final double tokenizeMs;
        public final double embeddingMs;
        public final double mixMs;
        public final double llmMs;
        public final double prefillMs;
        public final double decodeMs;
        public final int promptTokens;
        public final int generatedTokens;

        InferenceStats(JSONObject response) {
            totalMs = response.optLong("elapsed_ms");
            imageLoadMs = response.optDouble("image_load_ms");
            preprocessMs = response.optDouble("preprocess_ms");
            visionMs = response.optDouble("vision_ms");
            tokenizeMs = response.optDouble("tokenize_ms");
            embeddingMs = response.optDouble("embedding_ms");
            mixMs = response.optDouble("mix_ms");
            llmMs = response.optDouble("llm_ms");
            prefillMs = response.optDouble("prefill_ms");
            decodeMs = response.optDouble("decode_ms");
            promptTokens = response.optInt("prompt_tokens");
            generatedTokens = response.optInt("generated_tokens");
        }
    }

    public ModelRunner(Context context) {
        appContext = context.getApplicationContext();
        storage = new ModelStorage(appContext);
        if (storage.isReady()) {
            ContextCompat.startForegroundService(
                appContext, new Intent(appContext, VlmForegroundService.class));
        }
    }

    public boolean isReady() {
        return storage.isReady();
    }

    public void runText(String prompt, Callback callback) {
        runRequest(null, prompt, callback);
    }

    public void runVision(String imagePath, String prompt, Callback callback) {
        runRequest(imagePath, prompt, callback);
    }

    private void runRequest(String imagePath, String prompt, Callback callback) {
        executor.execute(() -> {
            try (LocalSocket socket = connectWithRetry();
                 DataOutputStream output = new DataOutputStream(socket.getOutputStream());
                 DataInputStream input = new DataInputStream(socket.getInputStream())) {
                JSONObject request = new JSONObject();
                request.put("version", 1);
                request.put("type", "generate");
                request.put("request_id", UUID.randomUUID().toString());
                request.put("prompt", prompt);
                request.put("image_path", imagePath == null ? "" : imagePath);
                writeFrame(output, request);

                while (true) {
                    JSONObject response = readFrame(input);
                    String type = response.optString("type");
                    if ("token".equals(type)) {
                        String chunk = response.optString("text");
                        mainHandler.post(() -> callback.onOutput(chunk));
                    } else if ("done".equals(type)) {
                        InferenceStats stats = new InferenceStats(response);
                        mainHandler.post(() -> callback.onComplete(stats));
                        return;
                    } else if ("error".equals(type)) {
                        throw new IOException(response.optString(
                            "message", "unknown server error"));
                    }
                }
            } catch (Exception exception) {
                String message = exception.getMessage();
                if (message == null || message.isEmpty()) {
                    message = exception.getClass().getSimpleName();
                }
                final String error = message;
                mainHandler.post(() -> callback.onError(error));
            }
        });
    }

    private LocalSocket connectWithRetry() throws Exception {
        long deadline = System.currentTimeMillis() + CONNECT_TIMEOUT_MS;
        Exception lastError = null;
        do {
            LocalSocket socket = new LocalSocket(LocalSocket.SOCKET_STREAM);
            try {
                socket.connect(new LocalSocketAddress(
                    INTERNAL_SOCKET_NAME, LocalSocketAddress.Namespace.ABSTRACT));
                return socket;
            } catch (Exception exception) {
                lastError = exception;
                try {
                    socket.close();
                } catch (IOException ignored) {
                }
                Thread.sleep(100);
            }
        } while (System.currentTimeMillis() < deadline);
        throw new IOException("VLM service not ready", lastError);
    }

    private static void writeFrame(DataOutputStream output, JSONObject message)
            throws Exception {
        byte[] payload = message.toString().getBytes(StandardCharsets.UTF_8);
        if (payload.length == 0 || payload.length > MAX_FRAME_SIZE) {
            throw new IOException("invalid request frame size: " + payload.length);
        }
        output.writeInt(payload.length);
        output.write(payload);
        output.flush();
    }

    private static JSONObject readFrame(DataInputStream input) throws Exception {
        final int size;
        try {
            size = input.readInt();
        } catch (EOFException exception) {
            throw new IOException("VLM service disconnected", exception);
        }
        if (size <= 0 || size > MAX_FRAME_SIZE) {
            throw new IOException("invalid response frame size: " + size);
        }
        byte[] payload = new byte[size];
        input.readFully(payload);
        return new JSONObject(new String(payload, StandardCharsets.UTF_8));
    }

    @Override
    public void close() {
        executor.shutdownNow();
    }
}
