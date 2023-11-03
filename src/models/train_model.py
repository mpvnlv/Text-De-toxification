from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import os
vocab_size = 150102  # Adjust as needed
vocab = {}
def collate_fn(batch):
    # print(batch)
    references = [sample['reference'] for sample in batch]
    translations = [sample['translation'] for sample in batch]
    length_diffs = [sample['lenght_diff'] for sample in batch]
    similarities = [sample['similarity'] for sample in batch]

    # Flatten the list of tokens and convert them to tensors
    all_reference_tokens = [token for reference_tokens in references for token in reference_tokens]
    all_translation_tokens = [token for translation_tokens in translations for token in translation_tokens]

    # Build the vocabulary from the tokens
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update({word: idx + len(vocab) for idx, word in enumerate(set(all_reference_tokens + all_translation_tokens))})

    # Convert tokens to indices and pad sequences to length 50
    references_padded = pad_sequence(
        [torch.tensor([vocab.get(token, vocab['<unk>']) for token in reference_tokens][:50]) for reference_tokens in references],
        batch_first=True,
        padding_value=vocab['<pad>']
    )

    translations_padded = pad_sequence(
        [torch.tensor([vocab.get(token, vocab['<unk>']) for token in translation_tokens][:50]) for translation_tokens in translations],
        batch_first=True,
        padding_value=vocab['<pad>']
    )

    return {
        'reference': references_padded,
        'translation': translations_padded,
        'lenght_diff': torch.tensor(length_diffs),
        'similarity': torch.tensor(similarities)
    }


def train(simple_paraphrase_model, train_dataset, optimizer, criterion, checkpoint_dir):
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0
        total_perplexity = 0  # Track cumulative perplexity

        for batch in train_dataloader:
          
            reference = batch['reference'].to('cuda')
            translation = batch['translation'].to('cuda')
            length_diff = batch['lenght_diff'].to('cuda')
            similarity = batch['similarity'].to('cuda')

            # Model forward pass
            output = simple_paraphrase_model(reference, length_diff)
            loss = criterion(output.view(-1, vocab_size), translation.view(-1))

            # Compute perplexity
            perplexity = torch.exp(loss)  # Using exponential to get perplexity
            total_loss += loss.item()
            total_perplexity += perplexity.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_dataloader)
        average_perplexity = total_perplexity / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Perplexity: {average_perplexity:.4f}')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': simple_paraphrase_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
            'perplexity': average_perplexity
        }

        checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')
# from torchtext.data.metrics import bleu_score

# def seq_to_text(sequence, vocab):
#     words = [vocab.itos[idx] for idx in sequence]  # Преобразование индексов в слова
#     text = ' '.join(words)  # Соединение слов в текстовую строку
#     return text

# def train(simple_paraphrase_model, train_dataset, optimizer, criterion, checkpoint_dir, max_length=50):
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

#     # Training loop
#     num_epochs = 10

#     for epoch in range(num_epochs):
#         total_loss = 0
#         total_bleu_score = 0  # Track cumulative BLEU score

#         for batch in train_dataloader:
#             reference = batch['reference'].to('cuda')
#             translation = batch['translation'].to('cuda')
#             length_diff = batch['lenght_diff'].to('cuda')
#             similarity = batch['similarity'].to('cuda')

#             # Model forward pass
#             output = simple_paraphrase_model(reference, length_diff, similarity)
#             loss = criterion(output.view(-1, vocab_size), translation.view(-1))

#             # Compute BLEU score
#             predicted = output.argmax(dim=2)
#             reference_text = [seq_to_text(reference[i], vocab) for i in range(reference.shape[0])]
#             predicted_text = [seq_to_text(predicted[i], vocab) for i in range(predicted.shape[0])]
#             bleu = bleu_score(predicted_text, reference_text, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
            
#             total_loss += loss.item()
#             total_bleu_score += bleu

#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         average_loss = total_loss / len(train_dataloader)
#         average_bleu_score = total_bleu_score / len(train_dataloader)
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, BLEU Score: {average_bleu_score:.4f}')
#         checkpoint = {
#             'epoch': epoch + 1,
#             'model_state_dict': simple_paraphrase_model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': average_loss,
#             'bleu_score': average_bleu_score
#         }

#         checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
#         torch.save(checkpoint, checkpoint_path)
#         print(f'Checkpoint saved at {checkpoint_path}')
