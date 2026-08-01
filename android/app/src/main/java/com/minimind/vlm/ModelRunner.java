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
        // Copy missing/updated files, including nested QNN graphs and runtime libs.
        if (new File(SRC_ROOT).isDirectory()) {
            bootstrap(ctx, localDir);
        }
        File qnnVision = new File(modelDir, "vision_qnn/vision.mnn");
        visionModel = (qnnVision.isFile() ? qnnVision
            : new File(modelDir, "vision_encode_proj.mnn")).getAbsolutePath();
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

            copyTree(srcModels, new File(modelDir));
        } catch (Exception e) {
            // Bootstrap failed — isReady() will report false
        }
    }

    private void copyTree(File src, File dst) throws Exception {
        if (src.isDirectory()) {
            dst.mkdirs();
            File[] children = src.listFiles();
            if (children != null) {
                for (File child : children) {
                    copyTree(child, new File(dst, child.getName()));
                }
            }
            return;
        }
        if (!dst.isFile() || dst.length() != src.length()
                || dst.lastModified() < src.lastModified()) {
            copyFile(src, dst);
            dst.setLastModified(src.lastModified());
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
                configureQnnEnvironment(pb);
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
                configureQnnEnvironment(pb);
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
            && (new File(modelDir, "qnn/llm.mnn").isFile()
                || new File(modelDir, "llm.mnn").isFile());
    }

    void configureQnnEnvironment(ProcessBuilder pb) {
        String libDir = new File(modelDir, "qnn/lib").getAbsolutePath();
        String oldLd = pb.environment().get("LD_LIBRARY_PATH");
        pb.environment().put("LD_LIBRARY_PATH",
            oldLd == null || oldLd.isEmpty() ? libDir : libDir + ":" + oldLd);
        pb.environment().put("ADSP_LIBRARY_PATH", libDir);
    }
}
