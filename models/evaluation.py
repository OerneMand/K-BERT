import torch
from tqdm import tqdm 

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def model_loop(self, model, optimizer, ds, loader, epochs, tet_loop, cuda):
    """Neatly packaged function for doing the training, evaluation and testing loop

    Arguments:
        model {transformers.BertModel} -- Model from transformers library to use for inference or learning
        optimizer {transformers.AdamW} -- AdamW just because its the only I knew the object name of, other optmization objects from transformers are valid too
        ds {torch.IterableDataset} -- IterableDataset, probably preferably KBertDataset
        loader {torch.DataLoader} -- DataLoader, proably preferably instantiated with load_from_path function from dataset.py
        epochs {int} -- Number of epochs
        tet_loop {str} -- Training, evaluation or testing loop
        cuda {bool} -- Whether CUDA ressources are available
    """    
    # For storing our loss and accuracy for plotting
    loss, accuracy = 0, 0
    accuracy_list = []
    loss_set = []

    if tet_loop != "train":
        epochs = 1

    # Set our model to mode specified by tet_loop
    if tet_loop = "train":
        model.train()
    else:
        model.eval()

    for epoch in range(epochs):
        # Training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds.reload_data()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        for step, batch in tqdm(enumerate(loader), desc = "Batches", position = 0, total = ds.length //  ds.batch_size):
            # Add batch to device
            batch = tuple(t.to(device) if t != None else None for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels = batch
            if tet_loop != "train":
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    logits = model(
                        b_input_ids, 
                        position_ids = b_soft_pos_emb, 
                        token_type_ids = b_segment_emb, 
                        attention_mask=b_attention_mask, 
                        encoder_attention_mask = b_visibility_matrix
                    )[0]
                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    with open("val_predictions.csv", "w") as file:
                        predwriter = csv.writer(file)
                        predwriter.writerow("pred, label\n")
                        predwriter.writerows(zip(np.argmax(logits, axis=1).flatten(), label_ids.flatten()))
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                    
                    eval_accuracy += tmp_eval_accuracy
                    eval_accuracy_list.append(tmp_eval_accuracy)
                    nb_eval_steps += 1
            
            loss, logits = model(
                b_input_ids, 
                position_ids = b_soft_pos_emb, 
                token_type_ids = b_segment_emb, 
                attention_mask = b_attention_mask, 
                encoder_attention_mask = b_visibility_matrix, 
                labels=b_labels
            )

            optimizer.zero_grad() # Gradients are accumulated, useful for RNN but not for BERT-man
            # Forward pass
            if cuda:
                loss_set.append(loss.item()) #item when training on cuda device
            else:
                loss_set.append(loss)
            
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            
            # Update tracking variables
            if cuda:
                tr_loss += loss.item()
            else:
                tr_loss += loss
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            accuracy_list.append(flat_accuracy(logits, label_ids))
            