import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader


def collate_fn_test(batch):
    references = [sample['reference'] for sample in batch]
    translations = [sample['translation'] for sample in batch]

    # Flatten the list of tokens and convert them to tensors
    all_reference_tokens = [token for reference_tokens in references for token in reference_tokens]
    all_translation_tokens = [token for translation_tokens in translations for token in translation_tokens]

    # Build the vocabulary from the tokens
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update(
        {word: idx + len(vocab) for idx, word in enumerate(set(all_reference_tokens + all_translation_tokens))})

    # Convert tokens to indices and pad sequences to length 50
    references_padded = pad_sequence(
        [torch.tensor([vocab.get(token, vocab['<unk>']) for token in reference_tokens][:50]) for reference_tokens in
         references],
        batch_first=True,
        padding_value=vocab['<pad>']
    )

    translations_padded = pad_sequence(
        [torch.tensor([vocab.get(token, vocab['<unk>']) for token in translation_tokens][:50]) for translation_tokens in
         translations],
        batch_first=True,
        padding_value=vocab['<pad>']
    )

    return {
        'reference': references_padded,
        'translation': translations_padded,
        #         'lenght_diff': torch.tensor(length_diffs),
        #         'similarity': torch.tensor(similarities)
    }


def test(simple_paraphrase_model, length_diff, criterion, vocab_size, val_dataset):
    # # Assuming you have a SimpleParaphraseModel and DataLoader
    # model = SimpleParaphraseModel(vocab_size, embedding_dim, hidden_dim)
    # criterion = nn.CrossEntropyLoss()  # You might need to adjust the loss function
    test_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_test)

    # Testing loop
    simple_paraphrase_model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation during testing
        for batch in test_dataloader:  # Assuming you have a DataLoader for testing data
            reference = batch['reference'].to('cuda')
            translation = batch['translation'].to('cuda')
            #         length_diff = batch['lenght_diff']
            #         similarity = batch['similarity']

            # Model forward pass
            output = simple_paraphrase_model(reference, length_diff)

            # Compute loss
            loss = criterion(output.view(-1, vocab_size), translation.view(-1))

            total_loss += loss.item()

    average_loss = total_loss / len(test_dataloader)
    print(f'Test Loss: {average_loss:.4f}')
