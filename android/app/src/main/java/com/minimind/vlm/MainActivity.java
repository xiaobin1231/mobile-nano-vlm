package com.minimind.vlm;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_PICK = 100;

    private RecyclerView chatList;
    private EditText input;
    private ImageButton sendBtn, attachBtn;
    private ChatAdapter adapter;
    private ModelRunner runner;

    private String pendingImagePath;  // image selected but not yet sent

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        chatList = findViewById(R.id.chat_list);
        input = findViewById(R.id.input);
        sendBtn = findViewById(R.id.btn_send);
        attachBtn = findViewById(R.id.btn_attach);

        adapter = new ChatAdapter();
        chatList.setLayoutManager(new LinearLayoutManager(this));
        chatList.setAdapter(adapter);

        runner = new ModelRunner(this);

        if (!runner.isReady()) {
            adapter.add(ChatMessage.system(
                "模型未就绪。请确保 /data/local/tmp/mobile-nano-vlm/ 下有:\n" +
                "  minimind_cli\n" +
                "  models/llm.mnn\n" +
                "  models/vision_encode_proj.mnn\n" +
                "首次启动会自动复制到应用目录"));
        } else {
            adapter.add(ChatMessage.system("MiniMind VLM 已就绪，可以开始对话。"));
        }

        // Send button
        sendBtn.setOnClickListener(v -> sendMessage());

        // Attach image button
        attachBtn.setOnClickListener(v -> pickImage());
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, REQUEST_IMAGE_PICK);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_IMAGE_PICK || resultCode != RESULT_OK
                || data == null || data.getData() == null) return;

        // Copy selected image to app-local storage
        Uri uri = data.getData();
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            File local = new File(getCacheDir(), "upload_" + System.currentTimeMillis() + ".jpg");
            try (FileOutputStream os = new FileOutputStream(local)) {
                byte[] buf = new byte[8192];
                int n;
                while ((n = is.read(buf)) != -1) os.write(buf, 0, n);
            }
            pendingImagePath = local.getAbsolutePath();
            input.setHint("输入问题（将使用已选图片）...");
            Toast.makeText(this, "图片已选择", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Toast.makeText(this, "读取图片失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void sendMessage() {
        String text = input.getText().toString().trim();
        if (TextUtils.isEmpty(text)) return;

        boolean hasImage = pendingImagePath != null;
        final String imagePath = pendingImagePath;
        pendingImagePath = null;
        input.setText("");
        input.setHint("输入问题...");

        // Show user message
        if (hasImage) {
            adapter.add(ChatMessage.userImage(text, imagePath));
        } else {
            adapter.add(ChatMessage.user(text));
        }

        // Placeholder for assistant
        adapter.add(ChatMessage.assistant("思考中..."));
        chatList.scrollToPosition(adapter.getItemCount() - 1);
        setInputEnabled(false);

        ModelRunner.Callback cb = new ModelRunner.Callback() {
            private final StringBuilder fullResponse = new StringBuilder();

            @Override
            public void onOutput(String text) {
                fullResponse.append(text).append("\n");
                runOnUiThread(() -> {
                    adapter.updateLast(fullResponse.toString().trim());
                    chatList.scrollToPosition(adapter.getItemCount() - 1);
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    adapter.updateLast("错误: " + error);
                    setInputEnabled(true);
                });
            }
        };

        // Run in background
        new Thread(() -> {
            try {
                // Synchronous: wait for full output
                if (hasImage) {
                    runVisionSync(imagePath, text, cb);
                } else {
                    runTextSync(text, cb);
                }
                runOnUiThread(() -> setInputEnabled(true));
            } catch (Exception e) {
                runOnUiThread(() -> {
                    adapter.updateLast("错误: " + e.getMessage());
                    setInputEnabled(true);
                });
            }
        }).start();
    }

    private void runVisionSync(String imagePath, String prompt, ModelRunner.Callback cb)
            throws Exception {
        ProcessBuilder pb = new ProcessBuilder(
            "/system/bin/linker64", runner.binaryPath, "vision", runner.modelDir,
            runner.visionModel, imagePath, prompt);
        pb.redirectErrorStream(true);
        Process p = pb.start();

        try (java.io.BufferedReader r = new java.io.BufferedReader(
                new java.io.InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = r.readLine()) != null) {
                cb.onOutput(line);
            }
        }
        int code = p.waitFor();
        if (code != 0) cb.onError("exit code " + code);
    }

    private void runTextSync(String prompt, ModelRunner.Callback cb) throws Exception {
        ProcessBuilder pb = new ProcessBuilder(
            "/system/bin/linker64", runner.binaryPath, "text", runner.modelDir, prompt);
        pb.redirectErrorStream(true);
        Process p = pb.start();

        try (java.io.BufferedReader r = new java.io.BufferedReader(
                new java.io.InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = r.readLine()) != null) {
                cb.onOutput(line);
            }
        }
        int code = p.waitFor();
        if (code != 0) cb.onError("exit code " + code);
    }

    private void setInputEnabled(boolean enabled) {
        input.setEnabled(enabled);
        sendBtn.setEnabled(enabled);
        attachBtn.setEnabled(enabled);
    }
}
