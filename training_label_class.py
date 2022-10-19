def stratified_split(df, target, val_percent=0.2):
    """
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    """
    classes = list(df[target].unique())
    train_idxs, val_idxs = [], []
    for c in classes:
        idx = list(df[df[target] == c].index)
        np.random.shuffle(idx)
        val_size = int(len(idx)*val_percent)
        val_idxs += idx[:val_size]
        train_idxs += idx[val_size:]
    return train_idxs, val_idxs

_, sample_idxs = stratified_split(clean_book, 'label', 0.1)

train_idxs, val_idxs = stratified_split(clean_book, 'label', val_percent = 0.2)
sample_train_idxs, sample_val_idxs = stratified_split(clean_book[clean_book.index.isin(sample_idxs)], 'label', val_percent = 0.2)

def test_stratified(df, col):
    """
    Analyzes the ratio of different classes in a categorical variable within a dataframe
    Inputs:
    - dataframe
    - categorical column to be analyzed
    Returns: None
    """
    classes = list(df[col].unique())
    
    for c in classes:
        print(f'Proportion of records with {c}: {len(df[df[col] == c])*1./len(df):0.2} ({len(df[df[col] == c])} / {len(df)})')
    print("----------------------")
    
test_stratified(clean_book, 'label')
test_stratified(clean_book[clean_book.index.isin(train_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(val_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(sample_train_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(sample_val_idxs)], 'label')

sampling = False

x_train = np.stack(clean_book[clean_book.index.isin(sample_train_idxs if sampling else train_idxs)]['desc_tokens'])
y_train = clean_book[clean_book.index.isin(sample_train_idxs if sampling else train_idxs)]['label'].apply(lambda x:classes.index(x))

x_val = np.stack(clean_book[clean_book.index.isin(sample_val_idxs if sampling else val_idxs)]['desc_tokens'])
y_val = clean_book[clean_book.index.isin(sample_val_idxs if sampling else val_idxs)]['label'].apply(lambda x:classes.index(x))

x_test = np.stack(clean_test['desc_tokens'])
y_test = clean_test['label'].apply(lambda x:classes.index(x))

# how many LSTM layers to use?
# speed-complexity trade off (set # of epochs and test how # layers changes accuracy)

# initialize model and add embedding layer
model = Sequential()
# decide on number of hidden nodes
model.add(Embedding(len(vocabulary)+1, output_dim = 200, input_length = max_desc_length))

parameters = {'vocab': vocabulary,
              'eval_batch_size': 30,
              'batch_size': 20,
              'epochs': 2,
              'dropout': 0.2,
              'optimizer': 'Adam',
              'loss': 'binary_crossentropy',
              'activation':'sigmoid'}

def bookLSTM(x_train, y_train, x_val, y_val, params):
  """ This model predicts fiction VS non-fiction from the book description """
    model = Sequential()
    model.name = "Book Model2"
    model.add(Embedding(len(params['vocab'])+1, output_dim = x_train.shape[1], input_length = x_train.shape[1]))
    model.add(LSTM(200, return_sequences = True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(200))
    model.add(Dense(1, activation = params['activation']))
    model.compile(loss = params['loss'],
              optimizer = params['optimizer'],
              metrics = ['accuracy'])
    print(model.summary())
    model.fit(x_train, 
          y_train,
          validation_data = (x_val, y_val),
          batch_size = params['batch_size'], 
          epochs = params['epochs'])
    results = model.evaluate(x_test, y_test, batch_size = params['eval_batch_size'])
    return model

BookModel = bookLSTM(x_train, y_train, x_val, y_val, parameters)
