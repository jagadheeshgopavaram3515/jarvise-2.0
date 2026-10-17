export type ConversationEvent = {
  id: number; kind: string; text: string; role: string;
  speech_id?: string; turn_id?: string; timestamp?: number;
};
type Delivery = 'responding' | 'speaking' | 'spoken' | 'interrupted' | 'failed';
type Part = { speech_id: string; text: string; delivery: Delivery; visible: string };
export type Message = ConversationEvent & { delivery?: Delivery; parts?: Part[] };

function refresh(message: Message) {
  if (!message.parts?.length) return;
  message.text = message.parts.map(part => part.text).join(' ');
  message.delivery = message.parts.some(part => part.delivery === 'speaking') ? 'speaking'
    : message.parts.some(part => part.delivery === 'responding') ? 'responding'
    : message.parts.some(part => part.delivery === 'interrupted') ? 'interrupted'
    : message.parts.some(part => part.delivery === 'failed') ? 'failed' : 'spoken';
}

export function applyConversationEvents(previous: Message[], events: ConversationEvent[]): Message[] {
  const messages = previous.map(message => ({ ...message, parts: message.parts?.map(part => ({ ...part })) }));
  for (const event of events) {
    if (event.kind === 'speech_interrupted') {
      for (const message of messages) {
        if (message.turn_id === event.turn_id) {
          for (const part of message.parts ?? []) {
            if (['responding', 'speaking'].includes(part.delivery)) part.delivery = 'interrupted';
          }
          refresh(message);
        }
      }
      continue;
    }
    if (['response', 'subtitle', 'speech_start', 'speech_progress', 'speech_end', 'speech_failed'].includes(event.kind) && event.text) {
      const existing = event.speech_id ? messages.find(message => message.parts?.some(part => part.speech_id === event.speech_id)) : undefined;
      const delivery = ['speech_start', 'speech_progress'].includes(event.kind) ? 'speaking' : event.kind === 'speech_end' ? 'spoken'
        : event.kind === 'speech_failed' ? 'failed' : 'responding';
      if (existing) {
        const part = existing.parts!.find(part => part.speech_id === event.speech_id)!;
        if (event.kind.startsWith('speech_')) part.delivery = delivery;
        if (event.kind === 'speech_progress' || event.kind === 'speech_end') part.visible = event.text;
        refresh(existing);
      } else if (event.kind === 'subtitle' || event.kind === 'response' || event.kind === 'speech_start') {
        const last = messages[messages.length - 1];
        const part: Part = { speech_id: event.speech_id ?? `legacy-${event.id}`, text: event.text, delivery,
          visible: event.speech_id ? '' : event.text };
        if (event.role !== 'user' && event.turn_id && last?.role === 'assistant' && last.turn_id === event.turn_id) {
          last.parts = [...(last.parts ?? []), part];
          refresh(last);
        } else {
          messages.push({ ...event, delivery: event.role === 'user' ? undefined : delivery,
            parts: event.role === 'user' ? undefined : [part] });
        }
      }
    }
  }
  return messages.slice(-200);
}
