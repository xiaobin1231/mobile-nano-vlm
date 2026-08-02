package com.minimind.vlm;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MiniMindVLM";
    private static final int REQUEST_IMAGE_PICK = 100;
    private static final int REQUEST_NOTIFICATIONS = 101;

    private RecyclerView chatList;
    private EditText input;
    private ImageButton sendBtn;
    private ImageButton attachBtn;
    private ChatAdapter adapter;
    private ModelRunner runner;
    private String pendingImagePath;

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
        requestNotificationPermission();

        if (!runner.isReady()) {
            adapter.add(ChatMessage.system(
                "模型未就绪。请先将 minimind_cli 和 QNN 模型部署到 "
                + "/data/local/tmp/mobile-nano-vlm/，再重新打开 App。"));
            setInputEnabled(false);
        } else {
            adapter.add(ChatMessage.system("模型资源已就绪，后台服务正在加载模型。"));
        }

        sendBtn.setOnClickListener(view -> sendMessage());
        attachBtn.setOnClickListener(view -> pickImage());

        String smokePrompt = getIntent().getStringExtra("qnn_smoke_prompt");
        if (!TextUtils.isEmpty(smokePrompt) && runner.isReady()) {
            String smokeImage = getIntent().getStringExtra("qnn_smoke_image");
            if (!TextUtils.isEmpty(smokeImage)) pendingImagePath = smokeImage;
            input.setText(smokePrompt);
            sendMessage();
        }
    }

    private void requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33
                && ContextCompat.checkSelfPermission(this,
                    Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                new String[] {Manifest.permission.POST_NOTIFICATIONS},
                REQUEST_NOTIFICATIONS);
        }
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, REQUEST_IMAGE_PICK);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode,
                                    @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_IMAGE_PICK || resultCode != RESULT_OK
                || data == null || data.getData() == null) return;

        Uri uri = data.getData();
        try (InputStream source = getContentResolver().openInputStream(uri)) {
            if (source == null) throw new IllegalStateException("无法打开图片");
            File local = new File(
                getCacheDir(), "upload_" + System.currentTimeMillis() + ".jpg");
            try (FileOutputStream destination = new FileOutputStream(local)) {
                byte[] buffer = new byte[8192];
                int count;
                while ((count = source.read(buffer)) != -1) {
                    destination.write(buffer, 0, count);
                }
            }
            pendingImagePath = local.getAbsolutePath();
            input.setHint("输入问题（将使用已选图片）...");
            Toast.makeText(this, "图片已选择", Toast.LENGTH_SHORT).show();
        } catch (Exception exception) {
            Toast.makeText(this, "读取图片失败: " + exception.getMessage(),
                Toast.LENGTH_SHORT).show();
        }
    }

    private void sendMessage() {
        String prompt = input.getText().toString().trim();
        if (TextUtils.isEmpty(prompt) || runner == null || !runner.isReady()) return;

        boolean hasImage = pendingImagePath != null;
        String imagePath = pendingImagePath;
        pendingImagePath = null;
        input.setText("");
        input.setHint("输入问题...");

        adapter.add(hasImage
            ? ChatMessage.userImage(prompt, imagePath)
            : ChatMessage.user(prompt));
        adapter.add(ChatMessage.assistant("思考中..."));
        chatList.scrollToPosition(adapter.getItemCount() - 1);
        setInputEnabled(false);

        ModelRunner.Callback callback = new ModelRunner.Callback() {
            private final StringBuilder response = new StringBuilder();

            @Override
            public void onOutput(String chunk) {
                response.append(chunk);
                adapter.updateLast(response.toString());
                chatList.scrollToPosition(adapter.getItemCount() - 1);
            }

            @Override
            public void onComplete(ModelRunner.InferenceStats stats) {
                if (response.length() == 0) adapter.updateLast("模型未生成文本");
                Log.i(TAG, "total=" + stats.totalMs
                    + "ms imageLoad=" + stats.imageLoadMs
                    + "ms preprocess=" + stats.preprocessMs
                    + "ms vision=" + stats.visionMs
                    + "ms tokenize=" + stats.tokenizeMs
                    + "ms embedding=" + stats.embeddingMs
                    + "ms mix=" + stats.mixMs
                    + "ms llm=" + stats.llmMs
                    + "ms prefill=" + stats.prefillMs
                    + "ms decode=" + stats.decodeMs
                    + "ms tokens=" + stats.promptTokens
                    + "+" + stats.generatedTokens);
                setInputEnabled(true);
            }

            @Override
            public void onError(String error) {
                adapter.updateLast("错误: " + error);
                setInputEnabled(true);
            }
        };

        if (hasImage) {
            runner.runVision(imagePath, prompt, callback);
        } else {
            runner.runText(prompt, callback);
        }
    }

    private void setInputEnabled(boolean enabled) {
        input.setEnabled(enabled);
        sendBtn.setEnabled(enabled);
        attachBtn.setEnabled(enabled);
    }

    @Override
    protected void onDestroy() {
        if (runner != null) runner.close();
        super.onDestroy();
    }
}
