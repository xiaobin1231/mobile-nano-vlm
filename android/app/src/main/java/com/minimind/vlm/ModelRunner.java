package com.minimind.vlm;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.nio.channels.FileChannel;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Runs minimind_cli binary in a background thread and returns the output.
 * On first launch, copies binary + models from /data/local/tmp/ to app-local storage.
 */
public class ModelRunner {
    // Source: where adb pushes the files
    private static final String SRC_ROOT = "/data/local/tmp/mobile-nano-vlm";

    final String binaryPath;
    final String modelDir;
    final String visionModel;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    public ModelRunner(Context ctx) {
        File localDir = new File(ctx.getFilesDir(), "minimind");
        localDir.mkdirs();

        binaryPath = new File(localDir, "minimind_cli").getAbsolutePath();
        modelDir = new File(localDir, "models").getAbsolutePath();
        visionModel = new File(modelDir, "vision_encode_proj.mnn").getAbsolutePath();

        // Auto-copy from /data/local/tmp/ on first launch
        if (!new File(binaryPath).canExecute()) {
            bootstrap(ctx, localDir);
        }
    }

    /** Copy binary + models from /data/local/tmp/ into app-local storage. */
    private void bootstrap(Context ctx, File localDir) {
        try {
            File srcBin = new File(SRC_ROOT, "minimind_cli");
            File srcModels = new File(SRC_ROOT, "models");

            // Copy binary
            File dstBin = new File(localDir, "minimind_cli");
            copyFile(srcBin, dstBin);
            dstBin.setExecutable(true);

            // Copy models directory
            new File(modelDir).mkdirs();
            if (srcModels.isDirectory()) {
                for (File f : srcModels.listFiles()) {
                    if (f.isFile()) {
                        copyFile(f, new File(modelDir, f.getName()));
                    }
                }
            }
        } catch (Exception e) {
            // Bootstrap failed — isReady() will report false
        }
    }

    private void copyFile(File src, File dst) throws Exception {
        try (FileInputStream fis = new FileInputStream(src);
             FileOutputStream fos = new FileOutputStream(dst);
             FileChannel in = fis.getChannel();
             FileChannel out = fos.getChannel()) {
            in.transferTo(0, in.size(), out);
        }
    }

    public interface Callback {
        void onOutput(String text);
        void onError(String error);
    }

    /** Vision + text inference. */
    public void runVision(String imagePath, String prompt, Callback cb) {
        executor.execute(() -> {
            try {
                ProcessBuilder pb = new ProcessBuilder(
                    "/system/bin/linker64", binaryPath, "vision", modelDir, visionModel, imagePath, prompt);
                pb.redirectErrorStream(true);
                Process p = pb.start();

                StringBuilder out = new StringBuilder();
                try (BufferedReader r = new BufferedReader(
                        new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = r.readLine()) != null) {
                        out.append(line).append("\n");
                        String partial = line;
                        mainHandler.post(() -> cb.onOutput(partial));
                    }
                }
                int code = p.waitFor();
                if (code != 0) {
                    mainHandler.post(() -> cb.onError("Exit code " + code + ": " + out));
                }
            } catch (Exception e) {
                mainHandler.post(() -> cb.onError(e.getMessage()));
            }
        });
    }

    /** Text-only inference. */
    public void runText(String prompt, Callback cb) {
        executor.execute(() -> {
            try {
                ProcessBuilder pb = new ProcessBuilder(
                    "/system/bin/linker64", binaryPath, "text", modelDir, prompt);
                pb.redirectErrorStream(true);
                Process p = pb.start();

                StringBuilder out = new StringBuilder();
                try (BufferedReader r = new BufferedReader(
                        new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = r.readLine()) != null) {
                        out.append(line).append("\n");
                        String partial = line;
                        mainHandler.post(() -> cb.onOutput(partial));
                    }
                }
                int code = p.waitFor();
                if (code != 0) {
                    mainHandler.post(() -> cb.onError("Exit code " + code + ": " + out));
                }
            } catch (Exception e) {
                mainHandler.post(() -> cb.onError(e.getMessage()));
            }
        });
    }

    public boolean isReady() {
        return new File(binaryPath).canExecute()
            && new File(modelDir, "llm.mnn").exists();
    }
}
