from unittest.mock import right

import numpy as np
import pandas as pd
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from sklearn.metrics import (confusion_matrix, classification_report, 
                           recall_score, precision_score, roc_curve, auc,
                           precision_recall_curve, cohen_kappa_score,
                           matthews_corrcoef, brier_score_loss)

ht=0
wt=0
samples=0
sample_count=20

save_dir = 'Data-attributes'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

sdir='PAN'
for d in ['train','test']:
    filepaths=[]
    labels=[]

    dpath = os.path.join(sdir,d)
    classlist=os.listdir(dpath)
    for klass in classlist:
        classpath=os.path.join(dpath,klass)
        flist = os.listdir(classpath)
        for i,f in enumerate(flist):
            fpath = os.path.join(classpath,f)
            if i < sample_count:
                img = plt.imread(fpath)
                ht +=img.shape[0]
                wt +=img.shape[1]
                samples += 1

            filepaths.append(fpath)
            labels.append(klass)
    Fseries = pd.Series(filepaths,name='filepaths')
    Lseries = pd.Series(labels,name='labels')
    if d == "train":
        df = pd.concat([Fseries,Lseries],axis=1)
    else:
        test_df = pd.concat([Fseries,Lseries],axis=1)

trsplit = 0.8
strat = df['labels']
train_df, valid_df = train_test_split(df, train_size=0.9,shuffle=True,random_state=123,stratify=strat)

plt.figure(figsize=(10, 6))
groups = df.groupby('labels')
counts = groups.size()
counts.plot(kind='bar')
plt.title('Dataset Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
plt.close()

print("train_df length:",len(train_df))
print("test_df length:",len(test_df))
print("valid_df length:",len(valid_df))

plt.figure(figsize=(8, 6))
sizes = [len(train_df), len(valid_df), len(test_df)]
labels = ['Train', 'Validation', 'Test']
plt.bar(labels, sizes)
plt.title('Dataset Split Sizes')
plt.ylabel('Number of Images')
plt.savefig(os.path.join(save_dir, 'dataset_split_sizes.png'))
plt.close()

classes = list(train_df['labels'].unique())
class_count=len(classes)
groups = df.groupby('labels')
print('{0:^30s} {1:^13s}'.format('CLASS','IMG COUNT'))
for label in train_df['labels'].unique():
    group = groups.get_group(label)
    samples = len(group)
    print('{0:^30s} {1:^13s}'.format(label,str(len(group))))

wave = wt/samples
have = ht/samples
aspect_ratio = have/wave

print('Average Image Height:',have,' Average Image Width:',wave,' Aspect Ratio:',aspect_ratio)
print()


def balance(train_df, max_samples, min_samples, column, working_dir, image_size):
    train_df = train_df.copy()
    # make directories to store augmented images
    aug_dir = os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in train_df['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)
    # create and store the augmented images
    total = 0
    gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                             height_shift_range=.2, zoom_range=.2)
    groups = train_df.groupby('labels') 
    for label in train_df['labels'].unique(): 
        group = groups.get_group(label) 
        sample_count = len(group) 
        if sample_count < max_samples: 
            aug_img_count = 0
            delta = max_samples - sample_count  # number of augmented images to create
            target_dir = os.path.join(aug_dir, label)  # define where to write the images
            aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=image_size,
                                              class_mode=None, batch_size=1, shuffle=False,
                                              save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                              save_format='jpg')
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
            total += aug_img_count
    print('Total Augmented images created= ', total)
    # create aug_df and merge with train_df to create composite training set ndf
    if total > 0:
        aug_fpaths = []
        aug_labels = []
        classlist = os.listdir(aug_dir)
        for klass in classlist:
            classpath = os.path.join(aug_dir, klass)
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries = pd.Series(aug_fpaths, name='filepaths')
        Lseries = pd.Series(aug_labels, name='labels')
        aug_df = pd.concat([Fseries, Lseries], axis=1)
        train_df = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)

    print(list(train_df['labels'].value_counts()))
    return train_df

max_samples=250
min_samples=0
column='labels'
working_dir='PAN'
img_size=(450,450)
train_df=balance(train_df,max_samples,min_samples,column,working_dir,img_size)

gan_dir = os.path.join(sdir, 'rs_aug')
if os.path.exists(gan_dir):
    print("正在加载GAN生成的训练数据...")
    filepaths = []
    labels = []
    
    classlist = os.listdir(gan_dir)
    for klass in classlist:
        classpath = os.path.join(gan_dir, klass)
        if os.path.isdir(classpath):
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                filepaths.append(fpath)
                labels.append(klass)
    
    if filepaths:
        Fseries = pd.Series(filepaths, name='filepaths')
        Lseries = pd.Series(labels, name='labels')
        gan_df = pd.concat([Fseries, Lseries], axis=1)
        
        train_df = pd.concat([train_df, gan_df], axis=0).reset_index(drop=True)
        print(f"添加了 {len(gan_df)} 个GAN生成的样本到训练集")
        print("最终训练集大小:", len(train_df))


batch_size=15
trgen = ImageDataGenerator(horizontal_flip=True,rotation_range=20, width_shift_range=.2,
                                  height_shift_range=.2, zoom_range=.2)
t_and_v_gen = ImageDataGenerator()
train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                   class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                   class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
length = len(test_df)
test_batch_size = sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=40],reverse=True)[0]
test_steps = int(length/test_batch_size)
test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                   class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)
classes = list(train_gen.class_indices.keys())
class_indices = list(train_gen.class_indices.values())
class_count = len(classes)
labels = test_gen.labels
print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps, ' number of classes : ', class_count)
print ('{0:^25s}{1:^12s}'.format('class name', 'class index'))
for klass, index in zip(classes, class_indices):
    print(f'{klass:^25s}{str(index):^12s}')


def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen) 
    plt.figure(figsize=(20, 20))
    length = len(labels)
    if length < 25: 
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()

show_image_samples(train_gen)

class DilatedSpatialAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_attention_heads=8, conv_kernel_size=3, dilation_rate=1):
        super().__init__()
        self._initialize_attention_params(hidden_dim, num_attention_heads)
        self._initialize_conv_params(conv_kernel_size, dilation_rate)
        self._build_conv_layer(hidden_dim)

    def _initialize_attention_params(self, hidden_dim, num_attention_heads):
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = hidden_dim // num_attention_heads
        self.attention_scale = self.attention_head_dim ** -0.5

    def _initialize_conv_params(self, conv_kernel_size, dilation_rate):
        self.conv_kernel_size = conv_kernel_size
        self.dilation_rate = dilation_rate

    def _build_conv_layer(self, hidden_dim):
        self.spatial_conv = tf.keras.layers.Conv2D(
            filters=hidden_dim,
            kernel_size=self.conv_kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
            groups=hidden_dim
        )

    def _reshape_to_multihead(self, tensor, batch_size, spatial_size):
        reshaped = tf.reshape(tensor, [
            batch_size, 
            spatial_size, 
            self.num_attention_heads, 
            self.attention_head_dim
        ])
        return tf.transpose(reshaped, [0, 2, 1, 3])

    def _process_qkv(self, query, key, value, batch_size, height, width):
        spatial_size = height * width

        query_multihead = self._reshape_to_multihead(query, batch_size, spatial_size)

        key_conv = self.spatial_conv(key)
        value_conv = self.spatial_conv(value)

        key_multihead = self._reshape_to_multihead(key_conv, batch_size, spatial_size)
        value_multihead = self._reshape_to_multihead(value_conv, batch_size, spatial_size)

        return query_multihead, key_multihead, value_multihead

    def _compute_attention(self, query, key, value):
        attention_scores = tf.matmul(query, key, transpose_b=True) * self.attention_scale
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        return tf.matmul(attention_weights, value)

    def _reshape_output(self, tensor, batch_size, height, width):
        output_transposed = tf.transpose(tensor, [0, 2, 1, 3])
        return tf.reshape(output_transposed, [
            batch_size, height, width, self.hidden_dim
        ])

    def call(self, inputs):
        # 解包输入
        query, key, value = inputs
        batch_size = tf.shape(query)[0]
        height = tf.shape(query)[1]
        width = tf.shape(query)[2]

        q_multi, k_multi, v_multi = self._process_qkv(
            query, key, value, batch_size, height, width
        )

        attended_values = self._compute_attention(q_multi, k_multi, v_multi)

        return self._reshape_output(attended_values, batch_size, height, width)

class MultiBranchFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=8):
        super(MultiBranchFeatureEnhancement, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self._init_normalization_layers()
        self._init_spatial_attention_layers()
        self._init_multiscale_conv_layers()
        self._init_fusion_layers()
        
    def _init_normalization_layers(self):
        self.horizontal_norm = tf.keras.layers.LayerNormalization()
        self.vertical_norm = tf.keras.layers.LayerNormalization()
        
    def _init_spatial_attention_layers(self):
        self.dilated_spatial_attn1 = DilatedSpatialAttention(
            hidden_dim=self.dim,
            num_attention_heads=self.num_heads//2,
            conv_kernel_size=3,
            dilation_rate=2
        )
        self.dilated_spatial_attn2 = DilatedSpatialAttention(
            hidden_dim=self.dim,
            num_attention_heads=self.num_heads//2,
            conv_kernel_size=3,
            dilation_rate=3
        )
        
    def _init_multiscale_conv_layers(self):
        self.kernel_sizes = {
            'small': 5,
            'medium': 7,
            'large': 11
        }
        
        self.horizontal_branches = {
            f'horizontal_{size}': tf.keras.layers.Conv2D(
                self.dim // 4, 
                (1, kernel_size), 
                padding='same', 
                groups=self.dim // 4
            ) for size, kernel_size in self.kernel_sizes.items()
        }
        
        self.vertical_branches = {
            f'vertical_{size}': tf.keras.layers.Conv2D(
                self.dim // 4, 
                (kernel_size, 1), 
                padding='same', 
                groups=self.dim // 4
            ) for size, kernel_size in self.kernel_sizes.items()
        }
    
    def _init_fusion_layers(self):
        self.feature_fusion = tf.keras.layers.Conv2D(self.dim, 1, 1)
        self.activation = tf.keras.layers.Activation('relu')
    
    def _process_directional_features(self, x, norm_layer, conv_branches):
        normalized = norm_layer(x)
        return [branch(normalized) for branch in conv_branches.values()]
    
    def _apply_spatial_attention(self, x):
        return [
            self.dilated_spatial_attn1([x, x, x]),
            self.dilated_spatial_attn2([x, x, x])
        ]
    
    def call(self, x, training=True):
        residual = x
        
        horizontal_features = self._process_directional_features(
            x, self.horizontal_norm, self.horizontal_branches
        )
        vertical_features = self._process_directional_features(
            x, self.vertical_norm, self.vertical_branches
        )
        
        attention_features = self._apply_spatial_attention(x)
        
        multi_scale_features = tf.concat(
            horizontal_features + vertical_features + attention_features,
            axis=-1
        )
        enhanced_features = self.feature_fusion(multi_scale_features)
        enhanced_features = self.activation(enhanced_features)
        
        return enhanced_features + residual

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ClinicalReasoningModule(tf.keras.layers.Layer):
    def __init__(self, channels, reasoning_steps=3, num_heads=4, dilation_rates=[2,3]):
        super().__init__()
        self.channels = channels
        self.reasoning_steps = reasoning_steps
        self.num_heads = num_heads
        self.dilation_rates = dilation_rates
        
        self.hypothesis_generator = tf.keras.Sequential([
            tf.keras.layers.Conv2D(channels, 1),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu')
        ])
        
        self.evidence_searchers = [
            DilatedSpatialAttention(
                hidden_dim=channels,
                num_attention_heads=num_heads,
                conv_kernel_size=3,
                dilation_rate=d
            ) for d in dilation_rates
        ]
        
        self.gate_fusion = tf.keras.Sequential([
            tf.keras.layers.Conv2D(channels, 1), 
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
        ])
        
        self.local_validator = tf.keras.Sequential([
            tf.keras.layers.Conv2D(channels//2, 1),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
        ])
        
        self.global_validator = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.alpha = self.add_weight(name='alpha', shape=(1,), initializer='zeros')
        
    def call(self, x, training=True):
        x_t = x
        for _ in range(self.reasoning_steps):
            hypothesis = self.hypothesis_generator(x_t)
        
            evidence = sum([searcher([x_t, x_t, x_t]) for searcher in self.evidence_searchers])
            evidence = tf.keras.activations.relu(evidence)
        
            concat_features = tf.concat([hypothesis, evidence], axis=-1)
            gate = self.gate_fusion(concat_features)
            fused = hypothesis * gate + evidence * (1 - gate)
            updated_hypothesis = self.hypothesis_generator(fused)
        
            gamma_local = self.local_validator(updated_hypothesis)
            gamma_global = self.global_validator(updated_hypothesis)
            
            gamma_global = tf.reshape(gamma_global, [-1, 1, 1, 1]) 
            
            gamma = self.alpha * gamma_local + (1 - self.alpha) * gamma_global
           
            x_t = gamma * x_t + (1 - gamma) * updated_hypothesis
            
        return x_t
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reasoning_steps": self.reasoning_steps,
            "num_heads": self.num_heads,
            "dilation_rates": self.dilation_rates
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def calculate_medical_metrics(y_true, y_pred, y_pred_prob):
    medical_metrics = {}
    
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)
    
    for i, class_name in enumerate(classes):
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)
        prob_pred = y_pred_prob[:, i]
        
        if len(np.unique(binary_true)) == 1 or len(np.unique(binary_pred)) == 1:
            sensitivity = 1.0 if np.all(binary_true == binary_pred) else 0.0
            specificity = 1.0 if np.all(binary_true == binary_pred) else 0.0
            ppv = 1.0 if np.all(binary_true == binary_pred) else 0.0
            npv = 1.0 if np.all(binary_true == binary_pred) else 0.0
        else:
            sensitivity = recall_score(binary_true, binary_pred)
            specificity = recall_score(~binary_true.astype(bool), ~binary_pred.astype(bool))
            ppv = precision_score(binary_true, binary_pred)
            npv = precision_score(~binary_true.astype(bool), ~binary_pred.astype(bool))
        
        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()
        dor = (tp * tn) / max(fp * fn, 1e-10) 
        
        youdens_index = sensitivity + specificity - 1
        
        fpr, tpr, _ = roc_curve(binary_true, prob_pred)
        auc_score = auc(fpr, tpr)
        precision_vals, recall_vals, _ = precision_recall_curve(binary_true, prob_pred)
        pr_auc = auc(recall_vals, precision_vals)
        
        medical_metrics[class_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'dor': dor,
            'youdens_index': youdens_index,
            'auc': auc_score,
            'pr_auc': pr_auc
        }
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'{class_name} - ROC')
        plt.legend()
        plt.savefig(os.path.join(medical_metrics_dir, f'roc_curve_{class_name}.png'))
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, label=f'PR-AUC = {pr_auc:.3f}')
        plt.xlabel('Rell')
        plt.ylabel('Precision')
        plt.title(f'{class_name} - PR')
        plt.legend()
        plt.savefig(os.path.join(medical_metrics_dir, f'pr_curve_{class_name}.png'))
        plt.close()
    
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    brier_scores = []
    for i in range(len(classes)):
        binary_true = (y_true == i)
        prob_pred = y_pred_prob[:, i]
        brier_scores.append(brier_score_loss(binary_true, prob_pred))
    
    medical_metrics['overall'] = {
        'kappa': kappa,
        'mcc': mcc,
        'mean_brier_score': np.mean(brier_scores)
    }
    
    return medical_metrics

img_shape = (img_size[0], img_size[1], 3)
base_model = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=img_shape,
    pooling=None  
)
base_model.trainable = True

x = base_model.output

x = tf.keras.layers.Conv2D(1536, 1, padding='same')(x)

x = MultiBranchFeatureEnhancement(dim=1536, num_heads=8)(x)

x = ClinicalReasoningModule(
    channels=1536, 
    reasoning_steps=3, 
    num_heads=4,
    dilation_rates=[2,3]
)(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(1024, kernel_regularizer=regularizers.l2(l=0.016),
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006),
          activation='relu')(x)
x = Dropout(rate=0.3, seed=123)(x)
x = Dense(128, kernel_regularizer=regularizers.l2(l=0.016),
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006),
          activation='relu')(x)
x = Dropout(rate=0.45, seed=123)(x)
output = Dense(class_count, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
with open('untitled.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
lr = 0.001
model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

save_dir = 'ZUIHONG'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")
    
medical_metrics_dir = os.path.join(save_dir, 'medical_metrics')
if not os.path.exists(medical_metrics_dir):
    os.makedirs(medical_metrics_dir)
    print(f"Created medical metrics directory: {medical_metrics_dir}")

model_save_path = os.path.join(save_dir, 'best_model.h5')
final_model_path = os.path.join(save_dir, 'final_model.h5')

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        model_save_path, 
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

steps_per_epoch = len(train_df) // batch_size
validation_steps = len(valid_df) // batch_size

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_gen,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_history.png'))
plt.close()

history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(save_dir, 'training_history.csv'))

test_loss, test_accuracy = model.evaluate(test_gen, steps=test_steps)
evaluation_results = f"""
测试集评估结果：
测试集准确率: {test_accuracy:.4f}
测试集损失: {test_loss:.4f}
"""

with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
    f.write(evaluation_results)

predictions = model.predict(test_gen, steps=test_steps)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes[:len(predicted_classes)]

medical_metrics = calculate_medical_metrics(true_classes, predicted_classes, predictions)

with open(os.path.join(medical_metrics_dir, 'detailed_medical_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("肺炎图像分类医学评估指标报告\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("各类别详细指标\n")
    f.write("-" * 30 + "\n")
    for class_name, metrics in medical_metrics.items():
        if class_name != 'overall':
            f.write(f"\n{class_name}类别的评估指标:\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
    
    f.write("\n整体模型评估指标\n")
    f.write("-" * 30 + "\n")
    for metric_name, value in medical_metrics['overall'].items():
        f.write(f"{metric_name}: {value:.4f}\n")

plt.figure(figsize=(12, 6))
class_names = [name for name in medical_metrics.keys() if name != 'overall']
metrics_to_plot = ['sensitivity', 'specificity', 'ppv', 'npv']
values = np.array([[medical_metrics[class_name][metric] for metric in metrics_to_plot] 
                   for class_name in class_names])

x = np.arange(len(class_names))
width = 0.2
multiplier = 0

for metric, metric_values in zip(metrics_to_plot, values.T):
    offset = width * multiplier
    plt.bar(x + offset, metric_values, width, label=metric)
    multiplier += 1

plt.xlabel('category')
plt.ylabel('Metric value')
plt.title('Comparison of indicators')
plt.xticks(x + width * 1.5, class_names, rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(medical_metrics_dir, 'metrics_comparison.png'))
plt.close()

print(f"\n医学评估指标已保存到 {medical_metrics_dir} 目录下")

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
plt.close()

class_report = classification_report(true_classes, predicted_classes, target_names=classes)
print("\nClassification Report:")
print(class_report)

with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write("分类报告：\n\n")
    f.write(class_report)

predictions_df = pd.DataFrame(predictions, columns=classes)
predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'))

with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

model.save(final_model_path)

print(f"\n所有评估指标和模型文件已保存到 {save_dir} 目录下")