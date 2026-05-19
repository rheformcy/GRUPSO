# ==========================================
    # 2. FUNGSI MODEL GRU STANDAR (MENGGUNAKAN PREPRO DAN SEED TERKUNCI)
    # ==========================================
    @st.cache_resource
    def jalankan_gru_standar(_df_emas):
        # Gunakan fungsi reset seed di awal fungsi
        reset_seeds()
        
        # --- PRAPEMROSESAN DATA (SAMA PERSIS DENGAN JALUR PSO) ---
        feature_cols = ["Terakhir"]
        target_col   = "Terakhir"
        data_features = _df_emas[feature_cols].values
        data_target = _df_emas[[target_col]].values

        n = len(data_features)
        n_train = int(n * 0.8)
        
        scaler_X = MinMaxScaler().fit(data_features[:n_train])
        scaler_y = MinMaxScaler().fit(data_target[:n_train])
        Xs = scaler_X.transform(data_features)
        ys = scaler_y.transform(data_target)

        def make_sequences(X_scaled, y_scaled, window=1):
            X_seq, y_seq = [], []
            for i in range(window, len(X_scaled)):
                X_seq.append(X_scaled[i-window:i])
                y_seq.append(y_scaled[i])
            return np.array(X_seq), np.array(y_seq)
    
        X_seq_all, y_seq_all = make_sequences(Xs, ys, window=1)
        dtrain_end = n_train - 1
        X_train = X_seq_all[:dtrain_end]
        y_train = y_seq_all[:dtrain_end]
        X_test = X_seq_all[dtrain_end:]
        y_test = y_seq_all[dtrain_end:]
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # --- MEMBANGUN ARSITEKTUR GRU ADAM STANDAR ---
        # Parameter baseline sesuai spesifikasi skripsimu
        GS_epoch = 50
        GS_batch = 32
        GS_units = 16
        GS_layers = 1
        GS_dropout = 0.0
        GS_LR = 0.001
        
        # Paksa Keras membersihkan sisa graf memori dan kunci seed tepat sebelum membuat model
        clear_session()
        tf.random.set_seed(SEED)
        
        gru_standar = Sequential()
        gru_standar.add(Input(shape=(1, 1))) # window=1, n_features=1
        gru_standar.add(GRU(units=GS_units, activation='tanh'))
        gru_standar.add(Dropout(GS_dropout))
        gru_standar.add(Dense(units=1, activation='linear'))
        
        gru_standar.compile(optimizer=Adam(learning_rate=GS_LR), loss='mse')
        
        # Eksekusi training dengan EarlyStopping bawaan
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        gru_standar.fit(
            X_train, y_train,
            epochs=GS_epoch,
            batch_size=GS_batch,
            callbacks=[early_stop],
            validation_split=0.2,
            verbose=1,
            shuffle=False # Menjaga agar urutan urutan gradien time series tetap konsisten
        )

        # --- EVALUASI METRIK ---
        y_pred_scaled = gru_standar.predict(X_test, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
        
        return GS_units, GS_LR, GS_batch, GS_dropout, rmse, mae, mape, y_test_inv, y_pred_inv
