def train(model, train_set, test_set, batch_size, epochs):
    x_train, y_train = train_set
    x_test, y_test = test_set
    history = {'training': {'accuracy': []}, 'test': {'accuracy': []}}
    for epoch in range(epochs):
        epoch_result = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=1,
            verbose=1
        )
        history['training']['accuracy'].append(epoch_result.history['acc'][0])
        y_hat = model.predict_classes(x_test)
        test_accuracy = len(y_hat[y_hat == y_test.ravel()]) / len(y_test)
        print('test acc:', test_accuracy)
        history['test']['accuracy'].append(test_accuracy)
    return history
