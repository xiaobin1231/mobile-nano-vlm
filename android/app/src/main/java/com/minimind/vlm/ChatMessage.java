package com.minimind.vlm;

/**
 * A single chat message.
 */
public class ChatMessage {
    public static final int TYPE_USER = 0;
    public static final int TYPE_ASSISTANT = 1;
    public static final int TYPE_SYSTEM = 2;

    public final int type;
    public String text;           // mutable for streaming updates
    public final String imagePath;  // non-null for user image messages

    public ChatMessage(int type, String text, String imagePath) {
        this.type = type;
        this.text = text;
        this.imagePath = imagePath;
    }

    public static ChatMessage user(String text) {
        return new ChatMessage(TYPE_USER, text, null);
    }

    public static ChatMessage userImage(String text, String imagePath) {
        return new ChatMessage(TYPE_USER, text, imagePath);
    }

    public static ChatMessage assistant(String text) {
        return new ChatMessage(TYPE_ASSISTANT, text, null);
    }

    public static ChatMessage system(String text) {
        return new ChatMessage(TYPE_SYSTEM, text, null);
    }
}
