import sys
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit,)
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMenu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer, confusion_matrix,mean_absolute_error,log_loss, hinge_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import tensorflow as tf
import time
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit
from tensorflow.keras import layers, models, optimizers

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        self.current_model_name = "None"

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    



    
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
                from tensorflow.keras.datasets import mnist
                from tensorflow.keras.utils import to_categorical

                (X_train, y_train), (X_test, y_test) = mnist.load_data()

                # Normalize
                X_train = X_train.astype("float32") / 255.0
                X_test = X_test.astype("float32") / 255.0

                # Reshape for Conv2D
                X_train = X_train.reshape(-1, 28, 28, 1)
                X_test = X_test.reshape(-1, 28, 28, 1)

                # One-hot encode labels
                y_train = to_categorical(y_train, 10)
                y_test = to_categorical(y_test, 10)

                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                self.status_bar.showMessage("MNIST loaded successfully.")
            
                return  # MNIST is handled separately
            elif dataset_name == "IMDB Sentiment":
                from tensorflow.keras.datasets import imdb
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                from tensorflow.keras.utils import to_categorical

                max_features = 10000  # sadece en sık geçen 10k kelime
                maxlen = 200          # cümle uzunluğu sabitlenecek

                (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

                # padding
                X_train = pad_sequences(X_train, maxlen=maxlen)
                X_test = pad_sequences(X_test, maxlen=maxlen)

                y_train = to_categorical(y_train, 2)
                y_test = to_categorical(y_test, 2)

                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                self.status_bar.showMessage("IMDB dataset loaded successfully.")
                return
            elif dataset_name == "Jena Climate Dataset":
                try:
 

                    zip_path = tf.keras.utils.get_file(
                        origin="https://storage.googleapis.com/download.tensorflow.org/data/jena_climate_2009_2016.csv.zip",
                        fname="jena_climate_2009_2016.csv.zip",
                        extract=True,
                    )
                    csv_path = zip_path.replace(".zip", "")
                    df = pd.read_csv(csv_path)

                    temp = df["T (degC)"].values

                    def create_sequences(data, window=144):
                        X, y = [], []
                        for i in range(len(data) - window):
                            X.append(data[i:i+window])
                            y.append(data[i+window])
                        return np.array(X), np.array(y)

                    X, y = create_sequences(temp, window=144)
                    X = X[..., np.newaxis]  # RNN için 3D

                    # Train-test split
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    self.X_train = X_train
                    self.X_test = X_test
                    self.y_train = y_train
                    self.y_test = y_test

                    self.status_bar.showMessage("Jena Climate dataset loaded successfully.")
                    return

                except Exception as e:
                    self.show_error(f"Error loading Jena dataset: {str(e)}")
                    return

            else:
                self.show_error("Only MNIST is supported in this template.")
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
    
    def apply_supervised_reduction(self):
        try:
            
            class_count = len(np.unique(self.y_train))
            n_components = min(class_count - 1, 2)  

            lda = LDA(n_components=n_components)
            X_lda = lda.fit_transform(self.X_train, self.y_train)


            score = silhouette_score(X_lda, self.y_train)


            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], c=self.y_train, cmap='viridis')
            self.figure.colorbar(scatter)
            ax.set_title(f"LDA Reduction - Silhouette Score: {score:.4f}")
            self.canvas.draw()

            self.status_bar.showMessage("Supervised reduction and visualization completed.")

        except Exception as e:
            self.show_error(f"Supervised Reduction Error: {str(e)}")
    
    
    def load_custom_data(self):
        """Load custom dataset from CSV file with sliding window for RNN"""
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

                    # Pencereleme fonksiyonu (sliding window)
                    def create_sequences(X_array, y_array, window_size=7):
                        Xs, ys = [], []
                        for i in range(len(X_array) - window_size):
                            Xs.append(X_array[i:i+window_size])
                            ys.append(y_array[i+window_size])
                        return np.array(Xs), np.array(ys)

                    # Numpy dizilerine çevir
                    X_np = X.values
                    y_np = y.values
                    X_np = X_np.astype(np.float32)
                    y_np = y_np.astype(np.float32)
                    window_size = 7  # isteğe bağlı, GUI üzerinden ayarlanabilir

                    X_seq, y_seq = create_sequences(X_np, y_np, window_size)

                    split_type = self.split_type_combo.currentText()
                    test_size = self.split_spin.value()
                    val_size = self.val_split_spin.value()

                    if split_type == "Train/Test":
                        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                            X_seq, y_seq, test_size=test_size, random_state=42)
                        self.X_val, self.y_val = None, None

                    else:  # Train/Val/Test
                        X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                            X_seq, y_seq, test_size=test_size, random_state=42)

                        val_ratio = val_size / (1 - test_size)

                        self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                            X_temp, y_temp, test_size=val_ratio, random_state=42)

                    if self.undersample_checkbox.isChecked():
                        self.apply_undersampling()

                    # Apply scaling if selected
                    self.apply_scaling()

                    self.status_bar.showMessage(f"Loaded custom dataset with window size {window_size}: {file_name}")

        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    
    
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()

        # ---- Row 1: Dataset & Preprocessing ----
        row1_layout = QHBoxLayout()

        # Missing value options
        self.missing_combo = QComboBox()
        self.missing_combo.addItems([
            "None",
            "Mean Imputation",
            "Interpolation",
            "Forward Fill",
            "Backward Fill"
        ])
        row1_layout.addWidget(QLabel("Missing Data:"))
        row1_layout.addWidget(self.missing_combo)

        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "California Housing Dataset",
            "MNIST Dataset",
            "IMDB Sentiment",
            "Jena Climate Dataset"

        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        row1_layout.addWidget(QLabel("Dataset:"))
        row1_layout.addWidget(self.dataset_combo)

        # Load button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        row1_layout.addWidget(self.load_btn)

        # Scaling method
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        row1_layout.addWidget(QLabel("Scaling:"))
        row1_layout.addWidget(self.scaling_combo)
        # Feature extraction button
        self.extract_features_btn = QPushButton("Extract Features")
        self.extract_features_btn.clicked.connect(self.extract_features_from_signal)
        row1_layout.addWidget(self.extract_features_btn)
        self.segment_size_spin = QSpinBox()
        self.segment_size_spin.setRange(10, 1000)
        self.segment_size_spin.setValue(200)
        row1_layout.addWidget(QLabel("Segment Size:"))
        row1_layout.addWidget(self.segment_size_spin)
        self.undersample_checkbox = QCheckBox("Use Undersampling")
        row1_layout.addWidget(self.undersample_checkbox)

        data_layout.addLayout(row1_layout)

        # ---- Row 2: Split & K-Fold ----
        row2_layout = QHBoxLayout()

        # Split type
        self.split_type_combo = QComboBox()
        self.split_type_combo.addItems(["Train/Test", "Train/Val/Test"])
        row2_layout.addWidget(QLabel("Split Type:"))
        row2_layout.addWidget(self.split_type_combo)

        # Test size
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.05)
        row2_layout.addWidget(QLabel("Test Set Ratio:"))
        row2_layout.addWidget(self.split_spin)

        # Validation size
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.15)
        row2_layout.addWidget(QLabel("Validation Ratio:"))
        row2_layout.addWidget(self.val_split_spin)

        # K-Fold CV
        self.use_kfold_checkbox = QCheckBox("Use K-Fold CV")
        self.use_kfold_checkbox.setChecked(False)
        row2_layout.addWidget(self.use_kfold_checkbox)

        self.k_fold_spin = QSpinBox()
        self.k_fold_spin.setRange(2, 20)
        self.k_fold_spin.setValue(5)
        self.k_fold_spin.setEnabled(False)
        row2_layout.addWidget(QLabel("K:"))
        row2_layout.addWidget(self.k_fold_spin)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Accuracy", "MSE", "RMSE", "MAE"])
        row2_layout.addWidget(QLabel("Evaluation Metric:"))
        row2_layout.addWidget(self.metric_combo)
        self.use_kfold_checkbox.stateChanged.connect(
            lambda _: self.k_fold_spin.setEnabled(self.use_kfold_checkbox.isChecked())
        )
        


        # Add to main layout
        data_layout.addLayout(row2_layout)
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""

        widget = QWidget()
        layout = QGridLayout(widget)



        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()

        loss_layout = QHBoxLayout()
        loss_label = QLabel("Loss Function:")
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["MSE", "MAE", "Huber"])
        loss_layout.addWidget(loss_label)
        loss_layout.addWidget(self.loss_combo)
        regression_layout.addLayout(loss_layout)
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",}
        )
        
        regression_layout.addWidget(lr_group)

        # Random Forest Regressor
        rf_group = self.create_algorithm_group(
            "Random Forest Regressor",
            {
                "n_estimators": "int",
                "max_depth": "int",
                "min_samples_split": "int"
            }
        )
        regression_layout.addWidget(rf_group)
        

     
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {
                "C": "double",
                "epsilon": "double",
                "kernel": ["linear", "rbf", "poly"]
            }
        )
        regression_layout.addWidget(svr_group)
      
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)
       

        regression_group.setLayout(regression_layout)

        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        # Classification Loss Selection

        clf_loss_layout = QHBoxLayout()
        clf_loss_label = QLabel("Loss Function:")
        self.classification_loss_combo = QComboBox()
        self.classification_loss_combo.addItems(["Cross-Entropy", "Hinge"])
        clf_loss_layout.addWidget(clf_loss_label)
        clf_loss_layout.addWidget(self.classification_loss_combo)
        classification_layout.addLayout(clf_loss_layout)

        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double"}
        )
        
        classification_layout.addWidget(nb_group)
        # Prior Selection Layout
        prior_layout = QHBoxLayout()
        prior_label = QLabel("Priors:")
        self.prior_combo = QComboBox()
        self.prior_combo.addItems(["Uniform", "Custom"])
        prior_layout.addWidget(prior_label)
        prior_layout.addWidget(self.prior_combo)

        # Input field for custom prior
        self.prior_input = QLineEdit()
        self.prior_input.setPlaceholderText("e.g., 0.3, 0.7")
        prior_layout.addWidget(self.prior_input)

        classification_layout.addLayout(prior_layout)
        
        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int"}
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest Classifier",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        return widget
    
    
    def create_dim_reduction_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)  # QVBoxLayout yerine QGridLayout

        # === PCA 
        pca_group = QGroupBox("PCA Settings")
        pca_layout = QVBoxLayout()
        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(1, 100)
        self.n_components_spin.setValue(2)
        self.whiten_checkbox = QCheckBox("Whiten")
        pca_layout.addWidget(QLabel("Number of Components:"))
        pca_layout.addWidget(self.n_components_spin)
        pca_layout.addWidget(self.whiten_checkbox)

        pca_reduce_btn = QPushButton("Apply PCA and Save")
        pca_reduce_btn.clicked.connect(self.compress_and_save_data)
        pca_layout.addWidget(pca_reduce_btn)

        pca_var_btn = QPushButton("Show Explained Variance")
        pca_var_btn.clicked.connect(self.apply_pca_and_plot_variance)
        pca_layout.addWidget(pca_var_btn)


        self.pca_vis_combo = QComboBox()
        self.pca_vis_combo.addItems(["2D", "3D"])
        pca_layout.addWidget(QLabel("Visualization Type:"))
        pca_layout.addWidget(self.pca_vis_combo)

        pca_vis_btn = QPushButton("PCA Visualization")
        pca_vis_btn.clicked.connect(self.plot_pca_scatter)
        pca_layout.addWidget(pca_vis_btn)
        
        manual_btn = QPushButton("Show Manual PCA (Eigen Decomposition)")
        manual_btn.clicked.connect(self.show_pca_manual_view)
        pca_layout.addWidget(manual_btn)


        pca_group.setLayout(pca_layout)

        # === t-SNE
        tsne_group = QGroupBox("t-SNE Settings")
        tsne_layout = QVBoxLayout()
        self.tsne_perplexity_spin = QDoubleSpinBox()
        self.tsne_perplexity_spin.setRange(5, 100)
        self.tsne_perplexity_spin.setValue(30)
        self.tsne_components_spin = QSpinBox()
        self.tsne_components_spin.setRange(2, 3)
        self.tsne_components_spin.setValue(2)
        tsne_layout.addWidget(QLabel("Perplexity:"))
        tsne_layout.addWidget(self.tsne_perplexity_spin)
        tsne_layout.addWidget(QLabel("Number of Components (2 or 3):"))
        tsne_layout.addWidget(self.tsne_components_spin)

        tsne_vis_btn = QPushButton("Apply t-SNE and Visualize")
        tsne_vis_btn.clicked.connect(self.apply_tsne)
        tsne_layout.addWidget(tsne_vis_btn)

        tsne_group.setLayout(tsne_layout)

        # === UMAP
        umap_group = QGroupBox("UMAP Settings")
        umap_layout = QVBoxLayout()
        self.umap_n_neighbors_spin = QSpinBox()
        self.umap_n_neighbors_spin.setRange(2, 200)
        self.umap_n_neighbors_spin.setValue(15)
        self.umap_min_dist_spin = QDoubleSpinBox()
        self.umap_min_dist_spin.setRange(0.0, 1.0)
        self.umap_min_dist_spin.setSingleStep(0.05)
        self.umap_min_dist_spin.setValue(0.1)
        self.umap_components_spin = QSpinBox()
        self.umap_components_spin.setRange(2, 3)
        self.umap_components_spin.setValue(2)
        umap_layout.addWidget(QLabel("n_neighbors:"))
        umap_layout.addWidget(self.umap_n_neighbors_spin)
        umap_layout.addWidget(QLabel("min_dist:"))
        umap_layout.addWidget(self.umap_min_dist_spin)
        umap_layout.addWidget(QLabel("n_components:"))
        umap_layout.addWidget(self.umap_components_spin)

        umap_vis_btn = QPushButton("Apply UMAP and Visualize")
        umap_vis_btn.clicked.connect(self.apply_umap)
        umap_layout.addWidget(umap_vis_btn)

        umap_reduce_btn = QPushButton("Apply UMAP and Save (Feature Reduction)")
        umap_reduce_btn.clicked.connect(self.save_umap_reduction)
        umap_layout.addWidget(umap_reduce_btn)

        umap_group.setLayout(umap_layout)

        # === LDA
        lda_group = QGroupBox("LDA (Supervised Reduction)")
        lda_layout = QVBoxLayout()
        lda_btn = QPushButton("Apply LDA and Visualize")
        lda_btn.clicked.connect(self.apply_lda)
        lda_layout.addWidget(lda_btn)
        lda_group.setLayout(lda_layout)

        # === K-Means
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        self.kmeans_n_clusters = QSpinBox()
        self.kmeans_n_clusters.setRange(1, 20)
        self.kmeans_n_clusters.setValue(3)
        self.kmeans_max_iter = QSpinBox()
        self.kmeans_max_iter.setRange(1, 1000)
        self.kmeans_max_iter.setValue(300)
        self.kmeans_n_init = QSpinBox()
        self.kmeans_n_init.setRange(1, 20)
        self.kmeans_n_init.setValue(10)
        kmeans_layout.addWidget(QLabel("n_clusters:"))
        kmeans_layout.addWidget(self.kmeans_n_clusters)
        kmeans_layout.addWidget(QLabel("max_iter:"))
        kmeans_layout.addWidget(self.kmeans_max_iter)
        kmeans_layout.addWidget(QLabel("n_init:"))
        kmeans_layout.addWidget(self.kmeans_n_init)

        kmeans_train_btn = QPushButton("Train K-Means Parameters")
        kmeans_train_btn.clicked.connect(self.apply_kmeans)
        kmeans_layout.addWidget(kmeans_train_btn)

        elbow_btn = QPushButton("Find Optimal k (Elbow Method)")
        elbow_btn.clicked.connect(self.apply_elbow_method)
        kmeans_layout.addWidget(elbow_btn)

        kmeans_group.setLayout(kmeans_layout)


        layout.addWidget(pca_group,   0, 0)
        layout.addWidget(tsne_group,  0, 1)
        layout.addWidget(umap_group,  1, 0)
        layout.addWidget(lda_group,   1, 1)
        layout.addWidget(kmeans_group, 2, 0, 1, 2)

        return widget


    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_visualization(self):
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        # Ana tab widget
        self.viz_tab_widget = QTabWidget()

        # --- Tab 1: Normal Visualization ---
        normal_viz_widget = QWidget()
        normal_viz_layout = QHBoxLayout(normal_viz_widget)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)

        normal_viz_layout.addWidget(self.canvas, stretch=3)
        normal_viz_layout.addWidget(self.metrics_text, stretch=1)
        self.viz_tab_widget.addTab(normal_viz_widget, "Plots & Metrics")

        # --- Tab 2: Confusion Matrix ---
        cm_widget = QWidget()
        cm_layout = QVBoxLayout(cm_widget)
        self.confusion_matrix_label = QLabel("Confusion Matrix will appear here after training.")
        self.confusion_matrix_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cm_layout.addWidget(self.confusion_matrix_label)
        self.viz_tab_widget.addTab(cm_widget, "Confusion Matrix")

        # --- Tab 3: Gradient Histogram ---
        gradient_widget = QWidget()
        gradient_layout = QVBoxLayout(gradient_widget)
        self.gradient_figure = Figure(figsize=(6, 4))
        self.gradient_canvas = FigureCanvas(self.gradient_figure)
        gradient_layout.addWidget(self.gradient_canvas)
        self.viz_tab_widget.addTab(gradient_widget, "Gradient Histogram")

        # Artık tabları ekledikten sonra widget'ı layout'a ekle
        viz_layout.addWidget(self.viz_tab_widget)
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    
    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group
    


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
            
            import seaborn as sns

            if len(np.unique(self.y_test)) <= 10:  # Sınıflandırma için
                cm = confusion_matrix(self.y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')

                # Eğer daha önce varsa eski canvas kaldır
                if hasattr(self, 'cm_canvas'):
                    self.cm_canvas.setParent(None)

                self.cm_canvas = FigureCanvas(fig)

                cm_tab_layout = self.confusion_matrix_label.parentWidget().layout()

                # Eski label'ı kaldır
                self.confusion_matrix_label.setParent(None)

                cm_tab_layout.addWidget(self.cm_canvas)
                self.cm_canvas.draw()

                # Confusion Matrix tabını aktif et
                self.viz_tab_widget.setCurrentIndex(1)
            self.metrics_text.setText(metrics_text)

            self.update_visualization(y_pred)

        except Exception as e:
            self.show_error(f"Model training error: {str(e)}")


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

    
    def apply_undersampling(self):
        try:
            from imblearn.under_sampling import RandomUnderSampler

            if self.X_train is None or self.y_train is None:
                return

            rus = RandomUnderSampler(random_state=42)
            self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
            self.status_bar.showMessage("Undersampling uygulandı. Veri dengelendi.")
        except Exception as e:
            self.show_error(f"Undersampling Error: {str(e)}")

    
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
    

    def show_pca_manual_view(self):
        try:
            import numpy as np
            

            cov_matrix = np.array([[5, 2], [2, 3]])
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            text = "Covariance Matrix Σ:\n"
            text += str(cov_matrix) + "\n\n"
            text += "Eigenvalues (λ):\n"
            text += str(np.round(eigenvalues, 4)) + "\n\n"
            text += "Eigenvectors (v):\n"
            text += str(np.round(eigenvectors, 4)) + "\n\n"
            text += "1D projection formula (approx):\n"
            text += "z = 0.850·x + 0.526·y\n"

            dialog = QDialog(self)
            dialog.setWindowTitle("Manual PCA: Eigen Decomposition")
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
    
    
    def apply_projection(self):
        try:
            method = self.projection_method_combo.currentText()
            perplexity = self.tsne_perplexity_spin.value()
            projection_type = self.tsne_projection_combo.currentText()
            n_components = 2 if projection_type == "2D" else 3

            if method == "PCA":
                projector = PCA(n_components=n_components)
                X_embedded = projector.fit_transform(self.X_train)

            elif method == "t-SNE":
                projector = TSNE(n_components=n_components, perplexity=perplexity, init='random', random_state=42)

            elif method == "UMAP":
                projector = umap.UMAP(n_components=n_components, random_state=42)


            X_embedded = projector.fit_transform(self.X_train)

            # --- Clustering Quality Score ---
            from sklearn.metrics import silhouette_score
            score = silhouette_score(X_embedded, self.y_train)


            
            if n_components == 2:
                fig = px.scatter(x=X_embedded[:, 0], y=X_embedded[:, 1], color=self.y_train)
            else:
                fig = px.scatter_3d(x=X_embedded[:, 0], y=X_embedded[:, 1], z=X_embedded[:, 2], color=self.y_train)

            fig.update_layout(title=f"{method} Projection (Silhouette Score: {score:.4f})")
            fig.show()

            self.status_bar.showMessage(f"{method} projection completed.")

        except Exception as e:
            self.show_error(f"Projection Error: {str(e)}")
    
    
    def _extract_features_array(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        features = []
        for row in X:
            feats = [
                np.mean(row),
                np.std(row),
                np.min(row),
                np.max(row),
                np.ptp(row),
                np.sqrt(np.mean(row**2))  # RMS
            ]
            features.append(feats)
        return np.array(features)
    
    
    def extract_features_from_signal(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV Dataset", "", "CSV Files (*.csv)")
            if not file_name:
                return

            df_raw = pd.read_csv(file_name)
            all_columns = list(df_raw.columns)

            dialog = ColumnSelectionDialog(all_columns, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_cols = dialog.get_selected_columns()
            else:
                self.status_bar.showMessage("Feature extraction cancelled.")
                return

            features = {}
            for col in selected_cols:
                data = df_raw[col].values
                features[f"{col}_mean"] = np.mean(data)
                features[f"{col}_std"] = np.std(data)
                features[f"{col}_min"] = np.min(data)
                features[f"{col}_max"] = np.max(data)
                features[f"{col}_rms"] = np.sqrt(np.mean(data**2))

            df_features = pd.DataFrame([features])

            save_path, _ = QFileDialog.getSaveFileName(self, "Save Feature CSV", "", "CSV Files (*.csv)")
            if save_path:
                df_features.to_csv(save_path, index=False)
                self.status_bar.showMessage(f"Öznitelikler çıkarıldı ve kaydedildi: {save_path}")
            else:
                self.status_bar.showMessage("Kaydetme iptal edildi.")

        except Exception as e:
            self.show_error(f"Öznitelik çıkarma hatası: {str(e)}")

    
    
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


    def handle_missing_values(self, df):
        method = self.missing_combo.currentText()

        if method == "None":
            return df
        elif method == "Mean Imputation":
            return df.fillna(df.mean(numeric_only=True))
        elif method == "Interpolation":
            return df.interpolate()
        elif method == "Forward Fill":
            return df.ffill()
        elif method == "Backward Fill":
            return df.bfill()
        
        return df


    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    
    def show_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(message)
        msg.setWindowTitle("Model Result")
        msg.exec()   

    
    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        self.layer_list_widget = QListWidget()
        mlp_layout.addWidget(self.layer_list_widget)

        # Sağ tıklama menüsü (isteğe bağlı)
        self.layer_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_list_widget.customContextMenuRequested.connect(self.show_layer_context_menu)

        # Learning rate
        lr_layout = QHBoxLayout()
        # Optimizer selection
        optimizer_layout = QHBoxLayout()
        optimizer_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        optimizer_layout.addWidget(self.optimizer_combo)
        layout.addLayout(optimizer_layout, 3, 0, 1, 2)

        # Early stopping checkbox
        self.early_stop_checkbox = QCheckBox("Use Early Stopping")
        self.early_stop_checkbox.setChecked(True)  # varsayılan açık
        layout.addWidget(self.early_stop_checkbox)


        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        self.lr_schedule_combo = QComboBox()
        self.lr_schedule_combo.addItems(["None", "Step Decay", "Exponential Decay"])
        layout.addWidget(QLabel("Learning Rate Schedule:"))
        layout.addWidget(self.lr_schedule_combo)
        

        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        self.plot_gradients_checkbox = QCheckBox("Show Gradient Histogram During Training")
        mlp_layout.addWidget(self.plot_gradients_checkbox)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model_architecture)
        layout.addWidget(save_btn)

        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model_architecture)
        layout.addWidget(load_btn)

        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)

        # Data Augmentation Section
        augment_group = QGroupBox("Data Augmentation")
        augment_layout = QVBoxLayout()

        self.augment_checkbox = QCheckBox("Use Data Augmentation")
        augment_layout.addWidget(self.augment_checkbox)

        # Rotation
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation Range:"))
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(0, 90)
        self.rotation_spin.setValue(20)
        rotation_layout.addWidget(self.rotation_spin)
        augment_layout.addLayout(rotation_layout)

        # Zoom
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom Range:"))
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.0, 1.0)
        self.zoom_spin.setSingleStep(0.05)
        self.zoom_spin.setValue(0.15)
        zoom_layout.addWidget(self.zoom_spin)
        augment_layout.addLayout(zoom_layout)

        # Flip
        self.flip_checkbox = QCheckBox("Enable Horizontal Flip")
        augment_layout.addWidget(self.flip_checkbox)

        augment_group.setLayout(augment_layout)
        layout.addWidget(augment_group, 1, 1)

        # === Pretrained Models ===
        pretrained_group = QGroupBox("Pretrained Models")
        pretrained_layout = QVBoxLayout()

        self.pretrained_combo = QComboBox()
        self.pretrained_combo.addItems(["None", "VGG16", "ResNet50"])
        pretrained_layout.addWidget(QLabel("Select Pretrained Model:"))
        pretrained_layout.addWidget(self.pretrained_combo)

        self.load_image_btn = QPushButton("Load Image Folder")
        self.load_image_btn.clicked.connect(self.load_image_folder_dataset)
        pretrained_layout.addWidget(self.load_image_btn)
        train_pre_btn = QPushButton("Train Pretrained Model")
        train_pre_btn.clicked.connect(self.train_pretrained_model)
        pretrained_layout.addWidget(train_pre_btn)

        pretrained_btn = QPushButton("Load Pretrained Model")
        pretrained_btn.clicked.connect(self.load_pretrained_model)
        pretrained_layout.addWidget(pretrained_btn)

        pretrained_group.setLayout(pretrained_layout)
        layout.addWidget(pretrained_group)

        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    def update_layer_display(self):
            text = "Current Layer Configuration:\n\n"
            for i, layer in enumerate(self.layer_config):
                text += f"{i+1}. {layer['type']} - {layer['params']}\n"
            self.layer_display.setText(text)
    
    def add_layer_dialog(self, existing_layer=None, index=None):
            """Open a dialog to add or edit a neural network layer"""
            dialog = QDialog(self)
            dialog.setWindowTitle("Add Neural Network Layer" if existing_layer is None else "Edit Layer")
            layout = QVBoxLayout(dialog)

            # Layer type selection
            type_layout = QHBoxLayout()
            type_label = QLabel("Layer Type:")
            type_combo = QComboBox()
            type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"," BatchNormalization",])
            type_layout.addWidget(type_label)
            type_layout.addWidget(type_combo)
            layout.addLayout(type_layout)

            # Parameters input
            params_group = QGroupBox("Layer Parameters")
            params_layout = QVBoxLayout()
            self.layer_param_inputs = {}

            def update_params():
                # Clear previous parameter widgets
                for i in reversed(range(params_layout.count())):
                    widget_to_remove = params_layout.itemAt(i).widget()
                    if widget_to_remove:
                        params_layout.removeWidget(widget_to_remove)
                        widget_to_remove.setParent(None) # Ensure it's properly deleted
                self.layer_param_inputs.clear()

                layer_type = type_combo.currentText()
                if layer_type == "Dense":
                    units_input = QSpinBox()
                    units_input.setRange(1, 4096)
                    units_input.setValue(128)
                    self.layer_param_inputs["units"] = units_input

                    activation_combo = QComboBox()
                    activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax", "linear"])
                    self.layer_param_inputs["activation"] = activation_combo

                    # ------- BURASI YENİ -------
                    l2_checkbox = QCheckBox("Use L2 Regularization")
                    l2_rate     = QDoubleSpinBox()
                    l2_rate.setRange(0.0001, 0.1)
                    l2_rate.setValue(0.001)
                    params_layout.addWidget(l2_checkbox)
                    params_layout.addWidget(QLabel("L2 Rate:"))
                    params_layout.addWidget(l2_rate)

                    self.layer_param_inputs["use_l2"] = l2_checkbox
                    self.layer_param_inputs["l2_rate"] = l2_rate
                    # --------------------------
                    
                    params_layout.addWidget(QLabel("Units:"))
                    params_layout.addWidget(units_input)
                    params_layout.addWidget(QLabel("Activation:"))
                    params_layout.addWidget(activation_combo)

                elif layer_type == "Conv2D":
                    filters_input = QSpinBox()
                    filters_input.setRange(1, 1024) # Increased range
                    filters_input.setValue(32)
                    self.layer_param_inputs["filters"] = filters_input

                    kernel_input = QLineEdit()
                    kernel_input.setText("3,3") # No space for consistency
                    self.layer_param_inputs["kernel_size"] = kernel_input
                    
                    conv_activation_combo = QComboBox()
                    conv_activation_combo.addItems(["relu", "sigmoid", "tanh", "linear"]) # Linear for no activation
                    conv_activation_combo.setCurrentText("relu") # Default to relu
                    self.layer_param_inputs["activation"] = conv_activation_combo

                    params_layout.addWidget(QLabel("Filters:"))
                    params_layout.addWidget(filters_input)
                    params_layout.addWidget(QLabel("Kernel Size (örn: 3,3):"))
                    params_layout.addWidget(kernel_input)
                    params_layout.addWidget(QLabel("Activation:"))
                    params_layout.addWidget(conv_activation_combo)

                    l2_checkbox = QCheckBox("Use L2 Regularization")
                    l2_rate = QDoubleSpinBox()
                    l2_rate.setRange(0.0001, 0.1)
                    l2_rate.setValue(0.001)
                    params_layout.addWidget(l2_checkbox)
                    params_layout.addWidget(QLabel("L2 Rate:"))
                    params_layout.addWidget(l2_rate)

                    self.layer_param_inputs["use_l2"] = l2_checkbox
                    self.layer_param_inputs["l2_rate"] = l2_rate

                elif layer_type == "Dropout":
                    rate_input = QDoubleSpinBox()
                    rate_input.setRange(0.0, 0.9) # Max 0.9 is more common
                    rate_input.setSingleStep(0.1)
                    rate_input.setValue(0.5)
                    self.layer_param_inputs["rate"] = rate_input

                    params_layout.addWidget(QLabel("Dropout Rate:"))
                    params_layout.addWidget(rate_input)

                elif layer_type == "MaxPooling2D":
                    pool_label = QLabel("Pool Size (örn: 2,2):")
                    pool_input = QLineEdit()
                    pool_input.setText("2,2") # No space
                    self.layer_param_inputs["pool_size"] = pool_input

                    params_layout.addWidget(pool_label)
                    params_layout.addWidget(pool_input)

                elif layer_type == "BatchNormalization":
                    # BatchNormalization katmanının genelde parametreye ihtiyacı yoktur
                    notice_label = QLabel("Bu katmanın yapılandırma parametresi yok.")
                    params_layout.addWidget(notice_label)
                    self.layer_param_inputs.clear()
            
            type_combo.currentIndexChanged.connect(update_params)
            
            params_group.setLayout(params_layout)
            layout.addWidget(params_group)

            # Load existing layer values if in edit mode
            if existing_layer:
                type_combo.setCurrentText(existing_layer["type"])
                # Manually trigger update_params because currentIndexChanged might not fire if type is already set
                update_params() # Call update_params to ensure widgets for the type are created
                QApplication.processEvents() # Process events to ensure UI updates

                for param_name, value in existing_layer["params"].items():
                    widget = self.layer_param_inputs.get(param_name)
                    if widget:
                        if isinstance(widget, QSpinBox):
                            widget.setValue(int(value))
                        elif isinstance(widget, QDoubleSpinBox):
                            widget.setValue(float(value))
                        elif isinstance(widget, QComboBox):
                            idx = widget.findText(str(value))
                            if idx >= 0:
                                widget.setCurrentIndex(idx)
                        elif isinstance(widget, QLineEdit):
                            if isinstance(value, (tuple, list)):
                                widget.setText(",".join(map(str, value))) # Use comma without space
                            else:
                                widget.setText(str(value))
            else:
                update_params()  # Initial load of parameters for the default selected type

            # Buttons
            btn_layout = QHBoxLayout()
            add_btn = QPushButton("Update Layer" if existing_layer else "Add Layer")
            cancel_btn = QPushButton("Cancel")
            btn_layout.addWidget(add_btn)
            btn_layout.addWidget(cancel_btn)
            layout.addLayout(btn_layout)

            def save_layer():
                layer_type = type_combo.currentText()
                layer_params = {}
                try:
                    for param_name, widget in self.layer_param_inputs.items():
                        if isinstance(widget, QSpinBox):
                            layer_params[param_name] = widget.value()
                        elif isinstance(widget, QDoubleSpinBox):
                            layer_params[param_name] = widget.value()
                        elif isinstance(widget, QComboBox):
                            layer_params[param_name] = widget.currentText()
                        elif isinstance(widget, QLineEdit):
                            text_value = widget.text().replace(" ", "") # Remove spaces
                            if param_name in ["kernel_size", "pool_size"]:
                                # Ensure it's a tuple of two integers
                                parts = tuple(map(int, text_value.split(',')))
                                if len(parts) == 2:
                                    layer_params[param_name] = parts
                                else:
                                    raise ValueError(f"{param_name} iki tam sayıdan oluşmalıdır (örn: 3,3)")
                            else:
                                layer_params[param_name] = text_value # Store as string if not special handling
                except ValueError as e:
                    self.show_error(f"Parametre hatası: {e}")
                    return


                new_layer = {
                    "type": layer_type,
                    "params": layer_params
                }

                if index is not None: # Editing existing layer
                    self.layer_config[index] = new_layer
                else: # Adding new layer
                    self.layer_config.append(new_layer)

                dialog.accept()
                self.update_layer_display() # Make sure this updates the QListWidget

            add_btn.clicked.connect(save_layer)
            cancel_btn.clicked.connect(dialog.reject)
            dialog.exec()
            
    def show_layer_context_menu(self, position):
        menu = QMenu()
        edit_action = menu.addAction("Düzenle")
        delete_action = menu.addAction("Sil")
        action = menu.exec(self.layer_list_widget.mapToGlobal(position))

        selected_row = self.layer_list_widget.currentRow()
        if selected_row < 0:
            return

        if action == edit_action:
            layer = self.layer_config[selected_row]
            self.add_layer_dialog(existing_layer=layer, index=selected_row)  # 👈 index burada gönderilmeli
        elif action == delete_action:
            del self.layer_config[selected_row]
            self.update_layer_display()



    def update_layer_display(self):
        self.layer_list_widget.clear()
        for i, layer in enumerate(self.layer_config):
            text = f"{i+1}. {layer['type']} - {layer['params']}"
            self.layer_list_widget.addItem(QListWidgetItem(text))

    def delete_selected_layer(self):
        selected = self.layer_list_widget.currentRow()
        if selected >= 0:
            del self.layer_config[selected]
            self.update_layer_display()

    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setDecimals(6)  
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0001)

        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)

        
        group.setLayout(layout)
        return group
    def save_model_architecture(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "HDF5 files (*.h5);;JSON files (*.json)")
            if file_path:
                model = self.create_neural_network(input_shape=self.X_train.shape[1:])
                # Dummy compile to allow saving structure
                model.compile(optimizer='adam', loss='categorical_crossentropy')
                if file_path.endswith(".h5"):
                    model.save(file_path)
                elif file_path.endswith(".json"):
                    json_path = file_path
                    weights_path = file_path.replace(".json", ".weights.h5")
                    model_json = model.to_json()
                    with open(json_path, "w") as json_file:
                        json_file.write(model_json)
                    model.save_weights(weights_path)
                self.status_bar.showMessage(f"Model saved to: {file_path}")
        except Exception as e:
            self.show_error(f"Save Error: {str(e)}")

    def load_model_architecture(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "HDF5 files (*.h5);;JSON files (*.json)")
            if not file_path:
                return

            if file_path.endswith(".h5"):
                from tensorflow.keras.models import load_model
                model = load_model(file_path)
            elif file_path.endswith(".json"):
                from tensorflow.keras.models import model_from_json
                with open(file_path, "r") as json_file:
                    model_json = json_file.read()
                model = model_from_json(model_json)
                weights_path = file_path.replace(".json", ".weights.h5")
                model.load_weights(weights_path)

            self.loaded_model = model
            self.status_bar.showMessage(f"Model loaded from: {file_path}")
            self.metrics_text.append("✅ Model structure loaded.\n")

        except Exception as e:
            self.show_error(f"Load Error: {str(e)}")


    def load_pretrained_model(self):
        try:
            model_name = self.pretrained_combo.currentText()
            if model_name == "None":
                self.show_error("Please select a pretrained model.")
                return

            if not hasattr(self, "train_gen") or self.train_gen is None:
                self.show_error("Please load image data before loading pretrained model.")
                return

            input_shape = (224, 224, 3)
            from tensorflow.keras.applications import VGG16, ResNet50
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

            base_model = None
            if model_name == "VGG16":
                base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
            elif model_name == "ResNet50":
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

            base_model.trainable = False  # freeze convolutional base

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            x = Dense(128, activation='relu')(x)

            num_classes = self.train_gen.num_classes  # ✅ Doğru yerden alınmalı!
            predictions = Dense(num_classes, activation='softmax')(x)

            self.loaded_model = Model(inputs=base_model.input, outputs=predictions)
            self.loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            self.status_bar.showMessage(f"{model_name} loaded and ready for fine-tuning.")
            self.metrics_text.append(f"✅ {model_name} loaded.\n")

        except Exception as e:
            self.show_error(f"Pretrained Load Error: {str(e)}")


    
    def create_cnn_controls(self):
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()

        self.cnn_layer_config = []
        self.cnn_layer_list_widget = QListWidget()
        layout.addWidget(self.cnn_layer_list_widget)

        add_btn = QPushButton("Add CNN Layer")
        add_btn.clicked.connect(self.add_cnn_layer_dialog)
        layout.addWidget(add_btn)
        self.cnn_layer_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cnn_layer_list_widget.customContextMenuRequested.connect(self.show_cnn_layer_context_menu)

        group.setLayout(layout)
        return group
    def show_cnn_layer_context_menu(self, position):
        menu = QMenu()
        edit_action = menu.addAction("Düzenle")
        delete_action = menu.addAction("Sil")
        action = menu.exec(self.cnn_layer_list_widget.mapToGlobal(position))

        selected_row = self.cnn_layer_list_widget.currentRow()
        if selected_row < 0:
            return

        if action == edit_action:
            layer = self.cnn_layer_config[selected_row]
            self.edit_cnn_layer_dialog(existing_layer=layer, index=selected_row)
        elif action == delete_action:
            del self.cnn_layer_config[selected_row]
            self.update_cnn_layer_list()    
    def add_cnn_layer_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add CNN Layer")
        layout = QVBoxLayout(dialog)

        type_combo = QComboBox()
        type_combo.addItems(["Conv2D", "MaxPooling2D", "Flatten", "Dropout", "BatchNormalization"])
        layout.addWidget(QLabel("Layer Type:"))
        layout.addWidget(type_combo)

        params_layout = QVBoxLayout()
        self.cnn_param_inputs = {}

        def update_params():
            # Clear previous
            for i in reversed(range(params_layout.count())):
                widget = params_layout.itemAt(i).widget()
                if widget:
                    params_layout.removeWidget(widget)
                    widget.setParent(None)
            self.cnn_param_inputs.clear()

            layer_type = type_combo.currentText()

            if layer_type == "Conv2D":
                filters_spin = QSpinBox()
                filters_spin.setRange(1, 512)
                filters_spin.setValue(32)
                self.cnn_param_inputs["filters"] = filters_spin
                params_layout.addWidget(QLabel("Filters:"))
                params_layout.addWidget(filters_spin)

                kernel_input = QLineEdit("3,3")
                self.cnn_param_inputs["kernel_size"] = kernel_input
                params_layout.addWidget(QLabel("Kernel Size (e.g. 3,3):"))
                params_layout.addWidget(kernel_input)

                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "linear"])
                self.cnn_param_inputs["activation"] = activation_combo
                params_layout.addWidget(QLabel("Activation:"))
                params_layout.addWidget(activation_combo)

            elif layer_type == "MaxPooling2D":
                pool_input = QLineEdit("2,2")
                self.cnn_param_inputs["pool_size"] = pool_input
                params_layout.addWidget(QLabel("Pool Size (e.g. 2,2):"))
                params_layout.addWidget(pool_input)

            elif layer_type == "Dropout":
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 0.9)
                rate_input.setValue(0.5)
                self.cnn_param_inputs["rate"] = rate_input
                params_layout.addWidget(QLabel("Dropout Rate:"))
                params_layout.addWidget(rate_input)

            elif layer_type == "BatchNormalization":
                note = QLabel("No parameters.")
                params_layout.addWidget(note)

            elif layer_type == "Flatten":
                note = QLabel("No parameters.")
                params_layout.addWidget(note)

        type_combo.currentIndexChanged.connect(update_params)
        update_params()

        layout.addLayout(params_layout)
    
        def save():
            params = {}
            for key, widget in self.cnn_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    params[key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    params[key] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[key] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    val = widget.text().replace(" ", "")
                    if "," in val:
                        params[key] = tuple(map(int, val.split(",")))
                    else:
                        params[key] = val
            self.cnn_layer_config.append({
                "type": type_combo.currentText(),
                "params": params
            })
            self.update_cnn_layer_list()
            dialog.accept()

        save_btn = QPushButton("Add Layer")
        save_btn.clicked.connect(save)
        layout.addWidget(save_btn)

        dialog.setLayout(layout)
        dialog.exec()

    def update_cnn_layer_list(self):
        self.cnn_layer_list_widget.clear()
        for i, layer in enumerate(self.cnn_layer_config):
            text = f"{i+1}. {layer['type']} - {layer['params']}"
            self.cnn_layer_list_widget.addItem(QListWidgetItem(text))


    
    def create_rnn_controls(self):
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()

        self.rnn_layer_config = []
        self.rnn_layer_list_widget = QListWidget()
        layout.addWidget(self.rnn_layer_list_widget)

        # Butonlar
        add_btn = QPushButton("Add RNN Layer")
        add_btn.clicked.connect(self.add_rnn_layer_dialog)
        layout.addWidget(add_btn)
        self.rnn_layer_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.rnn_layer_list_widget.customContextMenuRequested.connect(self.show_rnn_layer_context_menu)   
        group.setLayout(layout)
        return group
    def show_rnn_layer_context_menu(self, position):
        menu = QMenu()
        edit_action = menu.addAction("Düzenle")
        delete_action = menu.addAction("Sil")
        action = menu.exec(self.rnn_layer_list_widget.mapToGlobal(position))

        selected_row = self.rnn_layer_list_widget.currentRow()
        if selected_row < 0:
            return

        if action == edit_action:
            layer = self.rnn_layer_config[selected_row]
            self.edit_rnn_layer_dialog(existing_layer=layer, index=selected_row)
        elif action == delete_action:
            del self.rnn_layer_config[selected_row]
            self.update_rnn_layer_list()
    
    def add_rnn_layer_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add RNN Layer")
        layout = QVBoxLayout(dialog)

        type_combo = QComboBox()
        type_combo.addItems(["LSTM", "GRU"])
        layout.addWidget(QLabel("Layer Type:"))
        layout.addWidget(type_combo)

        units_spin = QSpinBox()
        units_spin.setRange(1, 512)
        units_spin.setValue(64)
        layout.addWidget(QLabel("Units:"))
        layout.addWidget(units_spin)

        return_seq_checkbox = QCheckBox("Return Sequences")
        layout.addWidget(return_seq_checkbox)

        def save():
            self.rnn_layer_config.append({
                "type": type_combo.currentText(),
                "params": {
                    "units": units_spin.value(),
                    "return_sequences": return_seq_checkbox.isChecked()
                }
            })
            self.update_rnn_layer_list()
            dialog.accept()

        btn = QPushButton("Add Layer")
        btn.clicked.connect(save)
        layout.addWidget(btn)

        dialog.setLayout(layout)
        dialog.exec()

    def update_rnn_layer_list(self):
        self.rnn_layer_list_widget.clear()
        for i, layer in enumerate(self.rnn_layer_config):
            text = f"{i+1}. {layer['type']} - {layer['params']}"
            self.rnn_layer_list_widget.addItem(QListWidgetItem(text))

    
    def prepare_emg_data_for_rnn(self, window_size=20):
        # EMG verisinin self.X_train ve self.y_train formatında olduğunu varsayalım
        X = self.X_train  # shape: (num_samples, num_channels)
        y = self.y_train  # sınıf etiketi

        # Sliding window oluştur
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Etiketleri one-hot encode et (eğer çoklu sınıf varsa)
        from tensorflow.keras.utils import to_categorical
        num_classes = len(np.unique(y_seq))
        y_seq_cat = to_categorical(y_seq, num_classes=num_classes)

        return X_seq, y_seq_cat

    def load_image_folder_dataset(self):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Dataset Folder")
        if not folder_path:
            return

        img_size = (224, 224)
        batch_size = 32
        val_split = self.val_split_spin.value()

        datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)

        self.train_gen = datagen.flow_from_directory(
            folder_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training"
        )
        self.val_gen = datagen.flow_from_directory(
            folder_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation"
        )

        self.status_bar.showMessage(f"Loaded image dataset from: {folder_path}")



    def train_pretrained_model(self):
        try:
            # 1. Verinin yüklü olduğunu kontrol et
            if not hasattr(self, "train_gen") or self.train_gen is None:
                self.show_error("No image data loaded. Please load an image folder first.")
                return

            # 2. Modelin yüklü olduğunu kontrol et
            if not hasattr(self, "loaded_model") or self.loaded_model is None:
                self.show_error("No pretrained model loaded. Please select and load VGG16 or ResNet50.")
                return

            # 3. Temel eğitim parametrelerini al
            model = self.loaded_model
            input_shape = self.train_gen.image_shape
            num_classes = self.train_gen.num_classes
            epochs = self.epochs_spin.value()
            batch_size = self.batch_size_spin.value()
            learning_rate = self.lr_spin.value()

            # 4. Optimizer seçimi
            optimizer_name = self.optimizer_combo.currentText()
            optimizer_class = {
                "Adam": tf.keras.optimizers.Adam,
                "SGD": tf.keras.optimizers.SGD,
                "RMSprop": tf.keras.optimizers.RMSprop
            }.get(optimizer_name, tf.keras.optimizers.Adam)
            optimizer = optimizer_class(learning_rate=learning_rate)

            # 5. Modeli derle
            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            # 6. Callback'ler
            callbacks = [self.create_progress_callback()]
            if self.early_stop_checkbox.isChecked():
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True))
            if hasattr(self, "plot_gradients_checkbox") and self.plot_gradients_checkbox.isChecked():
                callbacks.append(GradientHistogramCallback(model, self.gradient_canvas, self.X_test, self.y_test))


            # 7. Eğitimi başlat
            history = model.fit(
                self.train_gen,
                validation_data=self.val_gen,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )

            # 8. Sonuçları GUI'ye yansıt
            self.history = history
            self.plot_training_history(history)

            self.status_bar.showMessage("✅ Pretrained model training complete.")
            self.metrics_text.append("✅ Fine-tuning complete.\n")

        except Exception as e:
            self.show_error(f"Pretrained Model Training Error:\n{str(e)}")

    
    def on_train_button_clicked(self):
        if hasattr(self, "loaded_model") and self.loaded_model:
            self.train_pretrained_model()
        else:
            self.train_neural_network()
    
    
    def train_neural_network(self):
        try:
            if self.X_train is None or self.y_train is None:
                self.show_error("No training data available.")
                return

            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()

            X_train = self.X_train.astype("float32")
            X_test = self.X_test.astype("float32")

            is_regression = len(np.unique(self.y_train)) > 20 and self.y_train.ndim == 1

            if is_regression:
                y_train = self.y_train.astype("float32")
                y_test = self.y_test.astype("float32")
                output_units = 1
                output_activation = "linear"
                loss_function = "mse"
                metrics = ["mae"]
            else:
                from sklearn.preprocessing import LabelEncoder
                from tensorflow.keras.utils import to_categorical

                encoder = LabelEncoder()
                y_train = encoder.fit_transform(self.y_train.astype("int"))
                y_test = encoder.transform(self.y_test.astype("int"))

                num_classes = len(np.unique(y_train))
                y_train = to_categorical(y_train, num_classes=num_classes)
                y_test = to_categorical(y_test, num_classes=num_classes)

                output_units = num_classes
                output_activation = "softmax"
                loss_function = "categorical_crossentropy"
                metrics = ["accuracy"]

            model = self.create_neural_network(input_shape=X_train.shape[1:])
            model.add(tf.keras.layers.Dense(output_units, activation=output_activation))

            optimizer_name = self.optimizer_combo.currentText()
            optimizer_class = {
                "Adam": tf.keras.optimizers.Adam,
                "SGD": tf.keras.optimizers.SGD,
                "RMSprop": tf.keras.optimizers.RMSprop
            }.get(optimizer_name, tf.keras.optimizers.Adam)
            optimizer = optimizer_class(learning_rate=learning_rate)

            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

            callbacks = [self.create_progress_callback()]
            if self.early_stop_checkbox.isChecked():
                callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))

            # ✅ Gradient Histogram (eğer checkbox seçiliyse)
            if hasattr(self, "plot_gradients_checkbox") and self.plot_gradients_checkbox.isChecked():
                callbacks.append(GradientHistogramCallback(model, self.gradient_canvas, self.X_test, self.y_test))
                self.viz_tab_widget.setCurrentIndex(2)  # Sekmeyi otomatik aç

            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks
            )

            self.history = history
            self.plot_training_history(history)

            if is_regression:
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                self.metrics_text.append(f"\n✅ Regression Evaluation:\nMSE: {mse:.4f}\nMAE: {mae:.4f}")
            else:
                y_pred = model.predict(X_test).argmax(axis=1)
                y_test_labels = y_test.argmax(axis=1)
                from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
                acc = accuracy_score(y_test_labels, y_pred)
                f1 = f1_score(y_test_labels, y_pred, average="macro")
                self.metrics_text.append(f"\n✅ Classification Evaluation:\nAccuracy: {acc:.4f}\nF1-Score: {f1:.4f}")

                # ✅ Confusion Matrix çiz
                import seaborn as sns
                import matplotlib.pyplot as plt
                cm = confusion_matrix(y_test_labels, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')

                if hasattr(self, 'cm_canvas'):
                    self.cm_canvas.setParent(None)

                self.cm_canvas = FigureCanvas(fig)

                cm_tab_layout = self.confusion_matrix_label.parentWidget().layout()
                self.confusion_matrix_label.setParent(None)
                cm_tab_layout.addWidget(self.cm_canvas)
                self.cm_canvas.draw()

                self.viz_tab_widget.setCurrentIndex(1)  # Confusion Matrix sekmesini aç

            self.status_bar.showMessage("Training completed.")

        except Exception as e:
            self.show_error(f"Training Error: {str(e)}")



    
    
    def create_neural_network(self, input_shape=None):
        """Create a Keras Sequential model using CNN, RNN, and Dense configurations"""
        from tensorflow.keras import layers, models, regularizers

        model = models.Sequential()

        # === CNN katmanları (varsa)
        for i, layer_conf in enumerate(self.cnn_layer_config):
            layer_type = layer_conf["type"]
            params = dict(layer_conf["params"])

            if "kernel_size" in params and isinstance(params["kernel_size"], str):
                params["kernel_size"] = tuple(map(int, params["kernel_size"].split(",")))
            if "pool_size" in params and isinstance(params["pool_size"], str):
                params["pool_size"] = tuple(map(int, params["pool_size"].split(",")))

            if 'use_l2' in params and params['use_l2']:
                params['kernel_regularizer'] = regularizers.l2(params.get('l2_rate', 0.001))

            if i == 0 and input_shape is not None and "input_shape" not in params:
                params["input_shape"] = input_shape

            if layer_type == "Conv2D":
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D(**params))
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
            elif layer_type == "BatchNormalization":
                model.add(layers.BatchNormalization())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())

        # === RNN katmanları (varsa)
        for i, layer_conf in enumerate(self.rnn_layer_config):
            layer_type = layer_conf["type"]
            params = dict(layer_conf["params"])

            if 'use_l2' in params and params['use_l2']:
                params['kernel_regularizer'] = regularizers.l2(params.get('l2_rate', 0.001))


            if i == 0 and len(input_shape) == 1:  # (samples, sequence_length)
                vocab_size = int(np.max(self.X_train)) + 1  # tüm kelime indekslerini kapsar
                embedding_dim = 128
                model.add(layers.Embedding(input_dim=vocab_size,
                                        output_dim=embedding_dim,
                                        input_length=input_shape[0]))
                input_shape = (input_shape[0], embedding_dim)  # embedding sonrası şekil

            # İlk katmana input shape ekle
            if i == 0 and input_shape is not None and "input_shape" not in params:
                params["input_shape"] = input_shape

            if layer_type == "LSTM":
                model.add(layers.LSTM(**params))
            elif layer_type == "GRU":
                model.add(layers.GRU(**params))

        
        # === Dense (MLP) katmanları
        for i, layer_conf in enumerate(self.layer_config):
            layer_type = layer_conf["type"]
            params = dict(layer_conf["params"])

            if 'use_l2' in params and params['use_l2']:
                params['kernel_regularizer'] = regularizers.l2(params.get('l2_rate', 0.001))

            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "BatchNormalization":
                model.add(layers.BatchNormalization())

        return model

    def create_sequences(self, X, y, window_size=20):
        Xs, ys = [], []
        for i in range(len(X) - window_size):
            Xs.append(X[i:i+window_size])
            ys.append(y[i+window_size])
        return np.array(Xs), np.array(ys)

    def create_progress_callback(self):
        """Create callback for progress bar + metrics text logging"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                super().__init__()
                self.gui = parent

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.gui.progress_bar.setValue(progress)

                # Günlük bilgi
                val_acc = logs.get('val_accuracy', 0)
                val_loss = logs.get('val_loss', 0)
                train_acc = logs.get('accuracy', 0)
                train_loss = logs.get('loss', 0)

                message = (
                    f"Epoch {epoch + 1}/{self.params['epochs']}:\n"
                    f"  Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f}\n"
                    f"  Val   Acc: {val_acc:.4f}, Loss: {val_loss:.4f}\n\n"
                )

                self.gui.metrics_text.append(message)

                # Sekmeyi değiştirmek istiyorsan GUI üzerinden yap
                self.gui.viz_tab_widget.setCurrentIndex(2)

            def on_train_end(self, logs=None):
                self.gui.metrics_text.append("✅ Training Completed.\n")
        
        return ProgressCallback(self)

        
    
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            
        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                
            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
        
        self.canvas.draw()
        
    
    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"
        
        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            
            metrics_text = f"Model: {self.current_model_name}\n\n"

            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = self.current_model.score(self.X_test, self.y_test)
            
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"
            
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)
        
        self.metrics_text.setText(metrics_text)
        
    
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()
  
    
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)


class ColumnSelectionDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Columns for Feature Extraction")
        self.selected_columns = []

        layout = QVBoxLayout()
        self.checkboxes = []

        for col in columns:
            checkbox = QCheckBox(col)
            checkbox.setChecked(True)  # Varsayılan: hepsi seçili
            layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        btn = QPushButton("Confirm")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

        self.setLayout(layout)

    def get_selected_columns(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]
    
class GradientHistogramCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, canvas, x_test=None, y_test=None, val_gen=None):
        super().__init__()
        self.model    = model
        self.canvas   = canvas
        self.x_test   = x_test
        self.y_test   = y_test
        self.val_gen  = val_gen

    def on_epoch_end(self, epoch, logs=None):
        # 1) Öncelikle generator varsa oradan batch çek
        if self.val_gen is not None:
            x_input, y_true = next(iter(self.val_gen))
        else:
            # 2) Aksi halde init ile gelen dizileri kullan
            x_input, y_true = self.x_test, self.y_test

        with tf.GradientTape() as tape:
            preds = self.model(x_input, training=True)
            loss  = self.model.compiled_loss(y_true, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        flat_grads = tf.concat(
            [tf.reshape(g, [-1]) for g in grads if g is not None],
            axis=0
        )
        ax.hist(flat_grads.numpy(), bins=50)
        ax.set_title(f"Gradient Histogram - Epoch {epoch + 1}")
        self.canvas.draw()



def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())




if __name__ == '__main__':
    main()

