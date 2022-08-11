def optimize(model, 
criterion, 
train_data, 
labels,
sam=True,
mixup_criterion=None, 
labels2=None,
mixup_lam=None,
distiller=None):
    if mixup_criterion is not None:
        assert len(labels2) == len(labels)
        assert mixup_lam is not None

    outputs = model(train_data)

    if mixup_criterion:
        train_loss = mixup_criterion(criterion, outputs, labels, labels2, mixup_lam)
    elif distiller:
        train_loss = distiller(train_data, labels)
    else:
        train_loss = criterion(outputs, labels)
    
    #Calculate batch accuracy and accumulate in epoch accuracy
    epoch_loss += train_loss / len(train_loader)
    output_labels = outputs.argmax(dim=1)
    train_acc = (output_labels == train_labels).float().mean()
    epoch_accuracy += train_acc / len(train_loader)

    if sam:
        train_loss.backward() #Gradient of loss
        optimizer.first_step(zero_grad=True) #Perturb weights
        outputs = vit(train_data) #Outputs based on perturbed weights
        if mixup_criterion:
            perturbed_loss = mixup_criterion(criterion, outputs, labels, labels2, mixup_lam)
        else:
            perturbed_loss = criterion(outputs, train_labels) #Loss with perturbed weights
        perturbed_loss.backward()#Gradient of perturbed loss
        optimizer.second_step(zero_grad=True) #Unperturb weights and updated weights based on perturbed losses
        optimizer.zero_grad() #Set gradients of optimized tensors to zero to prevent gradient accumulation
        iteration += 1
        progress_bar.update(1)

    else:
        # is_second_order attribute is added by timm on one optimizer
        # (adahessian)
        loss_scaler.scale(train_loss).backward(
            create_graph=(
                hasattr(optimizer, "is_second_order")
                and optimizer.is_second_order
            )
        )
        if optimizer_args.clip_grad is not None:
            # unscale the gradients of optimizer's params in-place
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                vit.parameters(), optimizer_args.clip_grad
            )

        n_accum += 1

        if n_accum == n_batch_accum:
            n_accum = 0
            loss_scaler.step(optimizer)
            loss_scaler.update()

            iteration += 1
            progress_bar.update(1)