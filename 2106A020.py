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
                # Önce test setini ayır
                X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                    X, data.target, test_size=test_size, random_state=42)

                # Validation oranı: kalan veriye göre hesaplanmalı
                val_ratio = val_size / (1 - test_size)

                self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                    X_temp, y_temp, test_size=val_ratio, random_state=42)

            if self.undersample_checkbox.isChecked():
                self.apply_undersampling()
            #___________________________________________________________
            # Apply scaling if selected
            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def apply_supervised_reduction(self):
        try:
            # LDA ile boyut indirgeme (en fazla class_count - 1 bileşen olabilir)
            class_count = len(np.unique(self.y_train))
            n_components = min(class_count - 1, 2)  # Görselleştirme için 2 ile sınırla

            lda = LDA(n_components=n_components)
            X_lda = lda.fit_transform(self.X_train, self.y_train)

            # Skor hesapla (örnek olarak Silhouette kullanıyoruz)
            score = silhouette_score(X_lda, self.y_train)

            # Görselleştir
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
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)

                # Ask user to select target column
                target_col = self.select_target_column(data.columns)

                if target_col:
                    X = data.drop(target_col, axis=1)
                    X = self.handle_missing_values(X)
                    y = data[target_col]

                    # Split parameters from GUI
                    split_type = self.split_type_combo.currentText()
                    test_size = self.split_spin.value()
                    val_size = self.val_split_spin.value()

                    if split_type == "Train/Test":
                        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                            X, y, test_size=test_size, random_state=42)
                        self.X_val, self.y_val = None, None

                    else:  # Train/Val/Test
                        # Önce test setini ayır
                        X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                            X, y, test_size=test_size, random_state=42)

                        # Validation oranı kalan veri üzerinden
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
            "MNIST Dataset"
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
        

#____________________________________________________        
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {
                "C": "double",
                "epsilon": "double",
                "kernel": ["linear", "rbf", "poly"]
            }
        )
        regression_layout.addWidget(svr_group)
#_____________________________________________________       
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)
       

        regression_group.setLayout(regression_layout)
#_________________________________________________________
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
#_______________________________________________________
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
#__________________________________________________________________
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

        # === PCA Bölümü ===
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

        # 2D/3D Görselleştirme Seçimi
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

        # === t-SNE Bölümü ===
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

        # === UMAP Bölümü ===
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

        # === LDA Bölümü ===
        lda_group = QGroupBox("LDA (Supervised Reduction)")
        lda_layout = QVBoxLayout()
        lda_btn = QPushButton("Apply LDA and Visualize")
        lda_btn.clicked.connect(self.apply_lda)
        lda_layout.addWidget(lda_btn)
        lda_group.setLayout(lda_layout)

        # === K-Means Bölümü ===
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

        # --- Yerleşim düzeni (grid) ekleme ---
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
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
        viz_layout.addWidget(self.canvas, stretch=3)         # %75 genişlik
        viz_layout.addWidget(self.metrics_text, stretch=1)   # %25 genişlik
    
    
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
    
    #_______________________________________________________

    def train_model(self, name, param_widgets):
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import make_scorer

            use_kfold = self.use_kfold_checkbox.isChecked()
            cv = self.k_fold_spin.value()
            selected_metric = self.metric_combo.currentText()

            model = None

            # MODELLER
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

# ====== K-FOLD CROSS VALIDATION KISMI ======
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


            # ====== NORMAL TRAIN/TEST EĞİTİM ======
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = model.predict(self.X_test)

            self.current_model_name = name
            self.current_model = model

            # METRİKLERİ HAZIRLAYALIM
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

            else:  # Classification modeller
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

            # METRİK PANELİNİ GÜNCELLE
            self.metrics_text.setText(metrics_text)

            # Görselleştirme ve diğer güncellemeler
            self.update_visualization(y_pred)

        except Exception as e:
            self.show_error(f"Model training error: {str(e)}")


    def plot_pca_scatter(self):
        try:
            vis_type = self.pca_vis_combo.currentText()  # Kullanıcının seçimi
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

            # X_train'den covariance matrix hesapla
            X = np.array(self.X_train)

            cov_matrix = np.cov(X, rowvar=False)  # Features üzerinden kovaryans
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Covariance Matrix ve Eigen Değerler Vektörler Text Hazırla
            text = "Covariance Matrix (Σ):\n"
            text += str(np.round(cov_matrix, 4)) + "\n\n"

            text += "Eigenvalues (λ):\n"
            text += str(np.round(eigenvalues, 4)) + "\n\n"

            text += "Eigenvectors (v):\n"
            text += str(np.round(eigenvectors, 4)) + "\n"

            # Sonucu göster
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

            # Güvenli eşleme
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

            # --- Plotly ile çizim ---
            
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

            # --- Silhouette score hesapla ---
            try:
                score = silhouette_score(X_embedded, self.y_train)
                title_text = f"t-SNE Projection (Silhouette Score: {score:.4f})"
            except Exception as e:
                title_text = "t-SNE Projection (Silhouette Score: N/A)"
                print("Silhouette Score Error (t-SNE):", e)

            # --- Plotly görselleştirme ---
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

            # PCA ile görselleştir
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
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    
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
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # One-hot encode target for classification
            y_train = tf.keras.utils.to_categorical(self.y_train)
            y_test = tf.keras.utils.to_categorical(self.y_test)
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            # Train model
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
    
    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        
        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))
                
        return model

           
    def train_neural_network(self):
        """Train the neural network"""
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = model.fit(self.X_train, self.y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(self.X_test, self.y_test),
                              callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
            
    
    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
        return ProgressCallback(self.progress_bar)
        
    
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

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

