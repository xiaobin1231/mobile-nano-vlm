package com.minimind.vlm;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Foreground owner of the resident native VLM process. */
public final class VlmForegroundService extends Service {
    private static final String TAG = "MiniMindVlmService";
    private static final String CHANNEL_ID = "minimind_vlm_runtime";
    private static final int NOTIFICATION_ID = 1001;

    private final ExecutorService daemonManager = Executors.newSingleThreadExecutor();
    private volatile boolean stopping;
    private volatile Process daemonProcess;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        Notification notification = buildNotification("正在加载端侧模型…");
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIFICATION_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
        daemonManager.execute(this::runDaemonLoop);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        stopping = true;
        Process process = daemonProcess;
        if (process != null) {
            process.destroy();
        }
        daemonManager.shutdownNow();
        super.onDestroy();
    }

    private void runDaemonLoop() {
        while (!stopping) {
            try {
                ModelStorage storage = new ModelStorage(this);
                if (!storage.isReady()) {
                    updateNotification("模型资源未就绪");
                    return;
                }

                ProcessBuilder builder = new ProcessBuilder(
                    "/system/bin/linker64",
                    storage.binaryPath,
                    "server",
                    storage.modelDir,
                    storage.visionModel,
                    "4");
                builder.directory(storage.localDir);
                builder.redirectErrorStream(true);
                storage.configureQnnEnvironment(builder);
                daemonProcess = builder.start();

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(daemonProcess.getInputStream()))) {
                    String line;
                    while (!stopping && (line = reader.readLine()) != null) {
                        Log.i(TAG, line);
                        if (line.startsWith("READY")) {
                            updateNotification("端侧模型已加载，等待请求");
                        }
                    }
                }
                int exitCode = daemonProcess.waitFor();
                daemonProcess = null;
                if (!stopping) {
                    Log.w(TAG, "native daemon exited: " + exitCode);
                    updateNotification("推理进程异常退出，正在重启…");
                    Thread.sleep(1000);
                }
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                return;
            } catch (Exception exception) {
                Log.e(TAG, "failed to run native daemon", exception);
                updateNotification("推理服务启动失败，正在重试…");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return;
        NotificationChannel channel = new NotificationChannel(
            CHANNEL_ID, "MiniMind VLM 推理服务", NotificationManager.IMPORTANCE_LOW);
        channel.setDescription("保持端侧视觉语言模型处于加载状态");
        getSystemService(NotificationManager.class).createNotificationChannel(channel);
    }

    private Notification buildNotification(String text) {
        Intent openApp = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
            this, 0, openApp, PendingIntent.FLAG_IMMUTABLE
                | PendingIntent.FLAG_UPDATE_CURRENT);
        Notification.Builder builder = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
            ? new Notification.Builder(this, CHANNEL_ID)
            : new Notification.Builder(this);
        return builder
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentTitle("MiniMind VLM")
            .setContentText(text)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build();
    }

    private void updateNotification(String text) {
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.notify(NOTIFICATION_ID, buildNotification(text));
    }
}
