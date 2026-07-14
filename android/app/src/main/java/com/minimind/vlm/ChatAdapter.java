package com.minimind.vlm;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.List;

public class ChatAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private final List<ChatMessage> messages = new ArrayList<>();

    @Override
    public int getItemViewType(int position) {
        return messages.get(position).type;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(parent.getContext());
        if (viewType == ChatMessage.TYPE_ASSISTANT || viewType == ChatMessage.TYPE_SYSTEM) {
            View v = inflater.inflate(R.layout.item_message_assistant, parent, false);
            return new AssistantHolder(v);
        } else {
            View v = inflater.inflate(R.layout.item_message_user, parent, false);
            return new UserHolder(v);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int pos) {
        ChatMessage msg = messages.get(pos);
        if (holder instanceof AssistantHolder) {
            ((AssistantHolder) holder).bind(msg);
        } else {
            ((UserHolder) holder).bind(msg);
        }
    }

    @Override
    public int getItemCount() {
        return messages.size();
    }

    public void add(ChatMessage msg) {
        messages.add(msg);
        notifyItemInserted(messages.size() - 1);
    }

    public void updateLast(String text) {
        if (messages.isEmpty()) return;
        int idx = messages.size() - 1;
        messages.get(idx).text = text;
        notifyItemChanged(idx);
    }

    // ---- View holders ----

    static class AssistantHolder extends RecyclerView.ViewHolder {
        TextView text;
        AssistantHolder(View v) {
            super(v);
            text = v.findViewById(R.id.text);
        }
        void bind(ChatMessage msg) {
            text.setText(msg.text);
        }
    }

    static class UserHolder extends RecyclerView.ViewHolder {
        TextView text;
        ImageView image;
        UserHolder(View v) {
            super(v);
            text = v.findViewById(R.id.text);
            image = v.findViewById(R.id.image);
        }
        void bind(ChatMessage msg) {
            text.setText(msg.text);
            if (msg.imagePath != null) {
                image.setVisibility(View.VISIBLE);
                Bitmap bm = BitmapFactory.decodeFile(msg.imagePath);
                if (bm != null) image.setImageBitmap(bm);
            } else {
                image.setVisibility(View.GONE);
            }
        }
    }
}
