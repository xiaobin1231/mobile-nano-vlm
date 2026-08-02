package com.minimind.vlm;

import android.content.Context;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.channels.FileChannel;

/** App-private binary/model paths shared by the UI client and VLM service. */
final class ModelStorage {
    private static final String SRC_ROOT = "/data/local/tmp/mobile-nano-vlm";

    final File localDir;
    final String binaryPath;
    final String modelDir;
    final String visionModel;

    ModelStorage(Context context) {
        localDir = new File(context.getFilesDir(), "minimind");
        localDir.mkdirs();
        binaryPath = new File(localDir, "minimind_cli").getAbsolutePath();
        modelDir = new File(localDir, "models").getAbsolutePath();

        File source = new File(SRC_ROOT);
        if (source.isDirectory()) {
            try {
                bootstrap(localDir);
            } catch (Exception ignored) {
                // isReady() reports incomplete resources to the UI/service.
            }
        }

        File qnnVision = new File(modelDir, "vision_qnn/vision.mnn");
        visionModel = (qnnVision.isFile() ? qnnVision
            : new File(modelDir, "vision_encode_proj.mnn")).getAbsolutePath();
    }

    boolean isReady() {
        return new File(binaryPath).canExecute()
            && new File(modelDir, "qnn/llm.mnn").isFile()
            && new File(modelDir, "vision_qnn/vision.mnn").isFile();
    }

    void configureQnnEnvironment(ProcessBuilder builder) {
        String libDir = new File(modelDir, "qnn/lib").getAbsolutePath();
        String oldLd = builder.environment().get("LD_LIBRARY_PATH");
        builder.environment().put("LD_LIBRARY_PATH",
            oldLd == null || oldLd.isEmpty() ? libDir : libDir + ":" + oldLd);
        builder.environment().put("ADSP_LIBRARY_PATH", libDir);
    }

    private void bootstrap(File destination) throws Exception {
        File sourceBinary = new File(SRC_ROOT, "minimind_cli");
        File sourceModels = new File(SRC_ROOT, "models");
        File destinationBinary = new File(destination, "minimind_cli");
        copyIfChanged(sourceBinary, destinationBinary);
        destinationBinary.setExecutable(true);
        copyTree(sourceModels, new File(modelDir));
    }

    private void copyTree(File source, File destination) throws Exception {
        if (source.isDirectory()) {
            destination.mkdirs();
            File[] children = source.listFiles();
            if (children != null) {
                for (File child : children) {
                    copyTree(child, new File(destination, child.getName()));
                }
            }
            return;
        }
        copyIfChanged(source, destination);
    }

    private void copyIfChanged(File source, File destination) throws Exception {
        if (!source.isFile()) return;
        if (destination.isFile() && destination.length() == source.length()
                && destination.lastModified() >= source.lastModified()) {
            return;
        }
        File parent = destination.getParentFile();
        if (parent != null) parent.mkdirs();
        try (FileInputStream input = new FileInputStream(source);
             FileOutputStream output = new FileOutputStream(destination);
             FileChannel inputChannel = input.getChannel();
             FileChannel outputChannel = output.getChannel()) {
            long offset = 0;
            while (offset < inputChannel.size()) {
                offset += inputChannel.transferTo(
                    offset, inputChannel.size() - offset, outputChannel);
            }
        }
        destination.setLastModified(source.lastModified());
    }
}
