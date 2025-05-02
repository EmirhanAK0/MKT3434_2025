# 📘 README – Dimensionality Reduction & Clustering GUI Project (2106A020)

This project delivers an interactive Python GUI application to support dimensionality reduction, clustering, and model validation methods. Below are the integrated features and corresponding core functions.

---

## 📊 PCA – Explained Variance and Visualization

- User can select `n_components`
- Visualizes cumulative explained variance
- Supports 2D/3D PCA projection
- Can export reduced data to CSV

```python
    def apply_pca_and_plot_variance(self):
        try:
            n_components = self.n_components_spin.value()
            whiten = self.whiten_checkbox.isChecked()

            pca = PCA(n_components=n_components, whiten=whiten)
            X_pca = pca.fit_transform(self.X_train)

            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = explained_variance.cumsum()

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
            ax.set_xlabel("Number of Components")
            ax.set_ylabel("Cumulative Explained Variance")
            ax.set_title("PCA Cumulative Explained Variance")
            ax.grid(True)

            self.canvas.draw()
            self.status_bar.showMessage("PCA and explained variance plot completed.")
        except Exception as e:
            self.show_error(f"PCA Error: {str(e)}")
    


    def plot_pca_scatter(self):
        try:
            vis_type = self.pca_vis_combo.currentText() 
            n_components = 3 if vis_type == "3D" else 2
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(self.X_train)

            if vis_type == "2D":
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train, cmap='tab10', s=10)
                plt.colorbar(scatter)
                plt.title("PCA Projection (2D)")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            elif vis_type == "3D":
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=self.y_train, cmap='tab10', s=10)
                fig.colorbar(scatter)
                ax.set_title("PCA Projection (3D)")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            self.show_error(f"PCA Scatter Error: {str(e)}")

    

    def compress_and_save_data(self):
        try:
            if self.X_train is None or self.y_train is None:
                self.show_error("No data loaded.")
                return

            n_components = self.n_components_spin.value()
            whiten = self.whiten_checkbox.isChecked()

            pca = PCA(n_components=n_components, whiten=whiten)
            X_pca = pca.fit_transform(self.X_train)

            df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])


            y_series = pd.Series(self.y_train).reset_index(drop=True)[:len(df)]
            df['target'] = y_series

            file_path, _ = QFileDialog.getSaveFileName(self, "Save PCA Result", "", "CSV Files (*.csv)")
            if file_path:
                df.to_csv(file_path, index=False)
                self.status_bar.showMessage(f"PCA result saved to: {file_path}")
        except Exception as e:
            self.show_error(f"PCA Save Error: {str(e)}")



```

---

## 🧠 Manual PCA Eigen Decomposition (Σ = [[5, 2], [2, 3]])

- Fixed matrix eigenvalue and eigenvector calculation
- Shown via popup as a theoretical demonstration

```python
    def show_pca_manual_view(self):
        try:
            import numpy as np


            X = np.array(self.X_train)

            cov_matrix = np.cov(X, rowvar=False) 
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)


            text = "Covariance Matrix (Σ):\n"
            text += str(np.round(cov_matrix, 4)) + "\n\n"

            text += "Eigenvalues (λ):\n"
            text += str(np.round(eigenvalues, 4)) + "\n\n"

            text += "Eigenvectors (v):\n"
            text += str(np.round(eigenvectors, 4)) + "\n"


            dialog = QDialog(self)
            dialog.setWindowTitle("PCA Manual View")

            layout = QVBoxLayout()
            text_edit = QTextEdit()
            text_edit.setPlainText(text)
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)

            dialog.setLayout(layout)
            dialog.resize(600, 400)
            dialog.exec()

        except Exception as e:
            self.show_error(f"PCA Manual View Error: {str(e)}")
    
    
```

---

## 📉 t-SNE – 2D/3D Projection with Silhouette Score

- Configurable `perplexity` and number of components
- Visualized via Plotly (scatter or 3D)
- Calculates silhouette score and displays in title
- Can export result as CSV

```python
    def apply_tsne(self):
        try:
            perplexity = self.tsne_perplexity_spin.value()
            dims = self.tsne_components_spin.value()
            tsne = TSNE(n_components=dims, perplexity=perplexity, init='random', random_state=42)
            X_embedded = tsne.fit_transform(self.X_train)

            # --- Silhouette score
            try:
                score = silhouette_score(X_embedded, self.y_train)
                title_text = f"t-SNE Projection (Silhouette Score: {score:.4f})"
            except Exception as e:
                title_text = "t-SNE Projection (Silhouette Score: N/A)"
                print("Silhouette Score Error (t-SNE):", e)

            # --- Plotly
            if dims == 2:
                fig = px.scatter(x=X_embedded[:, 0], y=X_embedded[:, 1], color=self.y_train)
            else:
                fig = px.scatter_3d(x=X_embedded[:, 0], y=X_embedded[:, 1], z=X_embedded[:, 2], color=self.y_train)

            fig.update_layout(title=title_text)
            fig.show()

        except Exception as e:
            self.show_error(f"t-SNE Error: {str(e)}")

      
    
```

---

## 📈 UMAP – Feature Reduction and Visualization

- Parameters: `n_neighbors`, `min_dist`, `n_components`
- Interactive 2D/3D scatter plot
- Can save reduced feature set

```python
    def apply_umap(self):
        try:
            n_neighbors = self.umap_n_neighbors_spin.value()
            min_dist = self.umap_min_dist_spin.value()
            n_components = self.umap_components_spin.value()

            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
            X_umap = reducer.fit_transform(self.X_train)

            if n_components == 2:
                fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], color=self.y_train)
            else:
                fig = px.scatter_3d(x=X_umap[:, 0], y=X_umap[:, 1], z=X_umap[:, 2], color=self.y_train)

            fig.update_layout(title="UMAP Projection")
            fig.show()

        except Exception as e:
            self.show_error(f"UMAP Error: {str(e)}")
    
    

    def save_umap_reduction(self):
        try:
            n_neighbors = self.umap_n_neighbors_spin.value()
            min_dist = self.umap_min_dist_spin.value()
            n_components = self.umap_components_spin.value()

            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
            X_umap = reducer.fit_transform(self.X_train)

            df = pd.DataFrame(X_umap, columns=[f"Dim{i+1}" for i in range(n_components)])
            df['target'] = pd.Series(self.y_train).reset_index(drop=True)[:len(df)]

            file_path, _ = QFileDialog.getSaveFileName(self, "Save UMAP Features", "", "CSV Files (*.csv)")
            if file_path:
                df.to_csv(file_path, index=False)
                self.status_bar.showMessage(f"UMAP feature data saved to: {file_path}")
        except Exception as e:
            self.show_error(f"UMAP Feature Save Error: {str(e)}")

    
```

---

## 🧬 LDA – Supervised Dimensionality Reduction

- Projects data to a 2D space using class labels
- Displays silhouette score for class separation
- Uses Plotly for colored scatter visualization

```python
    def apply_lda(self):
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
            from sklearn.metrics import silhouette_score
            lda = LDA(n_components=2)
            X_lda = lda.fit_transform(self.X_train, self.y_train)

            score = silhouette_score(X_lda, self.y_train)

            fig = px.scatter(x=X_lda[:, 0], y=X_lda[:, 1], color=self.y_train)
            fig.update_layout(title=f"LDA Projection (Silhouette Score: {score:.4f})")
            fig.show()

            self.status_bar.showMessage("LDA projection completed.")
        except Exception as e:
            self.show_error(f"LDA Error: {str(e)}")
    
```

---

## 🧪 K-Fold Cross-Validation (Accuracy, MSE, RMSE, MAE)

- User-selectable k-fold value
- Shows mean ± std for selected evaluation metric
- Supports multiple models (classifier/regressor)

```python
    def train_model(self, name, param_widgets):
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import make_scorer

            use_kfold = self.use_kfold_checkbox.isChecked()
            cv = self.k_fold_spin.value()
            selected_metric = self.metric_combo.currentText()

            model = None


            if name == "Linear Regression":
                model = LinearRegression(fit_intercept=param_widgets["fit_intercept"].isChecked())

            elif name == "Support Vector Regression":
                model = SVR(
                    C=param_widgets["C"].value(),
                    epsilon=param_widgets["epsilon"].value(),
                    kernel=param_widgets["kernel"].currentText()
                )

            elif name == "Logistic Regression":
                model = LogisticRegression(
                    C=param_widgets["C"].value(),
                    max_iter=param_widgets["max_iter"].value(),
                    multi_class=param_widgets["multi_class"].currentText()
                )

            elif name == "Support Vector Machine":
                model = SVC(
                    C=param_widgets["C"].value(),
                    kernel=param_widgets["kernel"].currentText(),
                    degree=param_widgets["degree"].value(),
                    probability=True
                )

            elif name == "Naive Bayes":
                var_smoothing = param_widgets["var_smoothing"].value()
                prior_mode = self.prior_combo.currentText()
                if prior_mode == "Uniform":
                    priors = None
                else:
                    try:
                        prior_text = self.prior_input.text()
                        priors = list(map(float, prior_text.strip().split(',')))
                    except:
                        self.show_error("Custom priors formatı yanlış. Örnek: 0.3, 0.7")
                        return
                model = GaussianNB(var_smoothing=var_smoothing, priors=priors)

            elif name == "Random Forest Regressor":
                model = RandomForestRegressor(
                    n_estimators=param_widgets["n_estimators"].value(),
                    max_depth=param_widgets["max_depth"].value() if param_widgets["max_depth"].value() > 0 else None,
                    min_samples_split=param_widgets["min_samples_split"].value(),
                    random_state=42
                )
            elif name == "Random Forest Classifier":
                model = RandomForestClassifier(
                    n_estimators=param_widgets["n_estimators"].value(),
                    max_depth=param_widgets["max_depth"].value() if param_widgets["max_depth"].value() > 0 else None,
                    min_samples_split=param_widgets["min_samples_split"].value(),
                    class_weight='balanced',

                    random_state=42
                )
            else:
                self.show_error("This model is not yet supported.")
                return

# ====== K-FOLD CROSS VALIDATION
            if use_kfold:
                if selected_metric == "Accuracy":
                    scoring = "accuracy"
                elif selected_metric == "MSE":
                    scoring = "neg_mean_squared_error"
                elif selected_metric == "RMSE":
                    scoring = make_scorer(lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)), greater_is_better=False)
                elif selected_metric == "MAE":
                    scoring = "neg_mean_absolute_error"
                else:
                    scoring = "accuracy"

                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)
                score_mean = np.abs(np.mean(scores))
                score_std = np.abs(np.std(scores))

                self.show_message(f"{name} - {cv}-Fold CV | {selected_metric}:\nMean: {score_mean:.4f} ± {score_std:.4f}")
                self.status_bar.showMessage(f"{name} | {cv}-Fold CV {selected_metric}: {score_mean:.4f} ± {score_std:.4f}")
                return


            # ====== NORMAL TRAIN/TEST 
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = model.predict(self.X_test)

            self.current_model_name = name
            self.current_model = model


            metrics_text = f"Model: {self.current_model_name}\n"
            metrics_text += f"Training Time: {elapsed_time:.4f} seconds\n\n"

            if name in ["Linear Regression", "Support Vector Regression", "Random Forest Regressor"]:  # Regression modeller
                loss_type = self.loss_combo.currentText()
                if loss_type == "MSE":
                    loss_value = mean_squared_error(self.y_test, y_pred)
                elif loss_type == "MAE":
                    loss_value = mean_absolute_error(self.y_test, y_pred)
                elif loss_type == "Huber":
                    loss_value = np.mean(np.where(
                        np.abs(self.y_test - y_pred) < 1.0,
                        0.5 * (self.y_test - y_pred) ** 2,
                        np.abs(self.y_test - y_pred) - 0.5
                    ))
                r2 = model.score(self.X_test, self.y_test)

                metrics_text += f"Loss ({loss_type}): {loss_value:.4f}\n"
                metrics_text += f"R² Score: {r2:.4f}\n"

            else: 
                loss_type = self.classification_loss_combo.currentText()
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test)
                    loss_value = log_loss(self.y_test, y_proba)
                else:
                    loss_value = 0.0

                accuracy = accuracy_score(self.y_test, y_pred)
                conf_matrix = confusion_matrix(self.y_test, y_pred)

                metrics_text += f"Accuracy: {accuracy:.4f}\n"
                metrics_text += f"Loss ({loss_type}): {loss_value:.4f}\n\n"
                metrics_text += "Confusion Matrix:\n"
                metrics_text += str(conf_matrix)


            self.metrics_text.setText(metrics_text)

            self.update_visualization(y_pred)

        except Exception as e:
            self.show_error(f"Model training error: {str(e)}")


```

---

## 📊 K-Means Clustering + Elbow Method

- User can set `n_clusters`, `max_iter`, and `n_init`
- Elbow plot used to determine optimal k
- Visualized in 2D (via PCA)
- Shows both inertia and silhouette score

```python
    def apply_kmeans(self):
        try:
            k = self.kmeans_n_clusters.value()
            max_iter = self.kmeans_max_iter.value()
            n_init = self.kmeans_n_init.value()

            kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init, random_state=42)
            cluster_labels = kmeans.fit_predict(self.X_train)

            inertia = kmeans.inertia_
            sil_score = silhouette_score(self.X_train, cluster_labels)

            # PCA visualization
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(self.X_train)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=cluster_labels, cmap='tab10', s=10)
            self.figure.colorbar(scatter)
            ax.set_title(f"K-Means Clustering (k={k})\nInertia: {inertia:.2f}, Silhouette: {sil_score:.2f}")
            self.canvas.draw()

            self.status_bar.showMessage("K-Means clustering completed.")
        except Exception as e:
            self.show_error(f"K-Means Error: {str(e)}")



    def apply_elbow_method(self):
        try:
            X = self.X_train
            inertias = []
            k_range = range(1, 11)  # K=1'den 10'a kadar dene

            for k in k_range:
                kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)

            # Plot inertia vs k
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(k_range, inertias, marker='o')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method for Optimal k")
            ax.grid(True)

            self.canvas.draw()
            self.status_bar.showMessage("Elbow method completed.")

        except Exception as e:
            self.show_error(f"Elbow Method Error: {str(e)}")
    
    
```

---

## 📂 Dataset Handling – Split & Scaling

- Supports `Train/Test` and `Train/Val/Test` splitting
- Optional undersampling
- Several scaling methods selectable (Standard, Min-Max, Robust)

```python
    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "California Housing Dataset":
                data = datasets.fetch_california_housing()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            X = pd.DataFrame(data.data)
            #___________________________________________________________
            X = self.handle_missing_values(X)

           
            
            split_type = self.split_type_combo.currentText()
            test_size = self.split_spin.value()
            val_size = self.val_split_spin.value()

            if split_type == "Train/Test":
                self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                    X, data.target, test_size=test_size, random_state=42)
                self.X_val, self.y_val = None, None  # Val seti yok

            else:  # Train/Val/Test
                
                X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                    X, data.target, test_size=test_size, random_state=42)

                
                val_ratio = val_size / (1 - test_size)

                self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                    X_temp, y_temp, test_size=val_ratio, random_state=42)

            if self.undersample_checkbox.isChecked():
                self.apply_undersampling()

            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    

    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:

                data = pd.read_csv(file_name)


                target_col = self.select_target_column(data.columns)

                if target_col:
                    X = data.drop(target_col, axis=1)
                    X = self.handle_missing_values(X)
                    y = data[target_col]


                    split_type = self.split_type_combo.currentText()
                    test_size = self.split_spin.value()
                    val_size = self.val_split_spin.value()

                    if split_type == "Train/Test":
                        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                            X, y, test_size=test_size, random_state=42)
                        self.X_val, self.y_val = None, None

                    else:  # Train/Val/Test
                        X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                            X, y, test_size=test_size, random_state=42)


                        val_ratio = val_size / (1 - test_size)

                        self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                            X_temp, y_temp, test_size=val_ratio, random_state=42)

                    if self.undersample_checkbox.isChecked():
                        self.apply_undersampling()

                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    
```

---

## 🧠 Signal-Based Feature Extraction

- Extracts 5 features from each EMG signal segment
- Segment size configurable
- Can apply class balancing via undersampling
- Saves extracted features to CSV

```python
    def extract_features_from_signal(self):
        try:
            import os
            from collections import defaultdict

            file_name, _ = QFileDialog.getOpenFileName(self, "Select Raw EMG CSV", "", "CSV Files (*.csv)")
            if not file_name:
                return

            df_raw = pd.read_csv(file_name)
            segment_size = self.segment_size_spin.value()
            target_column = "class"
            channels = [f"channel{i}" for i in range(1, 9)]

            if self.undersample_checkbox.isChecked():
                min_count = df_raw[target_column].value_counts().min()
                df_raw = df_raw.groupby(target_column).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
                self.status_bar.showMessage("Undersampling uygulandı.")

            features_list = []
            labels_list = []

            for i in range(0, len(df_raw) - segment_size + 1, segment_size):
                segment = df_raw.iloc[i:i+segment_size]
                label = segment[target_column].mode()[0]

                segment_features = []
                for ch in channels:
                    data = segment[ch].values
                    segment_features.extend([
                        np.mean(data),
                        np.std(data),
                        np.min(data),
                        np.max(data),
                        np.sqrt(np.mean(data**2))
                    ])

                features_list.append(segment_features)
                labels_list.append(label)

            # DataFrame oluştur
            column_names = [f"{ch}_{stat}" for ch in channels for stat in ['mean', 'std', 'min', 'max', 'rms']]
            df_features = pd.DataFrame(features_list, columns=column_names)
            df_features["class"] = labels_list

            save_path, _ = QFileDialog.getSaveFileName(self, "Save Features CSV", "", "CSV Files (*.csv)")
            if save_path:
                df_features.to_csv(save_path, index=False)
                self.status_bar.showMessage(f"Öznitelikler çıkarıldı ve kaydedildi: {save_path}")
            else:
                self.status_bar.showMessage("Kaydetme iptal edildi.")

        except Exception as e:
            self.show_error(f"Öznitelik çıkarma hatası: {str(e)}")
    
    
```
