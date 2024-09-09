import matplotlib
matplotlib.use('TkAgg')  # 在 import pyplot 之前设置
import matplotlib.pyplot as plt
from model import resnet50,resnet_18
from model_vgg import vgg
from model_efficientNet import efficient_net
from model_shuffleNet import shufflenet_v2
from model_goolenet import GoogLeNet
from Attention1 import mobilenet_v3_small,mobilenet_v3_large, SpatialAttention, create_feature_extraction_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.layers import Dropout,GlobalAveragePooling2D
import tensorflow as tf
import csv
import json
import os
import time
import glob
import random
import numpy as np
import seaborn as sns
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)

    data_root = "/home/sherlock/下载/图像分类网络/deep-learning-for-image-processing-master/data_set"  # get data root path
    image_path = os.path.join(data_root, "Fashion-MNIST")  # flower data set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    im_height = 224
    im_width = 224

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    batch_size = 16
    epochs = 50
    freeze_layer = False

    # class dict
    data_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
    class_num = len(data_class)
    
    class_dict = dict((value, index) for index, value in enumerate(data_class))

    # reverse value and key of dict
    inverse_dict = dict((val, key) for key, val in class_dict.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # load train images list
    random.seed(0)
    train_image_list = glob.glob(train_dir+"/*/*.png")
    random.shuffle(train_image_list)
    train_num = len(train_image_list)
    assert train_num > 0, "cannot find any .jpg file in {}".format(train_dir)
    train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]

    # load validation images list
    val_image_list = glob.glob(validation_dir+"/*/*.png")
    random.shuffle(val_image_list)
    val_num = len(val_image_list)
    assert val_num > 0, "cannot find any .jpg file in {}".format(validation_dir)
    val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    def process_train_img(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image,channels=1)
        # image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.repeat(image, repeats=3, axis=-1)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        image = tf.image.random_flip_left_right(image)
        # image = (image - 0.5) / 0.5
        image = image - 128
        #image = image - [_R_MEAN, _G_MEAN, _B_MEAN]
        return image, label

    def process_val_img(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image,channels=1)
        # image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.repeat(image, repeats=3, axis=-1)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        # image = (image - 0.5) / 0.5
        image = image - 128
        #image = image - [_R_MEAN, _G_MEAN, _B_MEAN]
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # load train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    print(train_dataset.shuffle(buffer_size=train_num)\
                                 .map(process_train_img, num_parallel_calls=AUTOTUNE))
    train_dataset = train_dataset.shuffle(buffer_size=train_num)\
                                 .map(process_train_img, num_parallel_calls=AUTOTUNE)\
                                 .repeat().batch(batch_size).prefetch(AUTOTUNE)
    

    # load train dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_val_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                             .repeat().batch(batch_size)

    # 实例化模型
    '''
    #resnet50
    feature = resnet50(num_classes=10, include_top=False)
    
    pre_weights_path = './pretrain_weights.ckpt'
    assert len(glob.glob(pre_weights_path + "*")), "cannot find {}".format(pre_weights_path)
    feature.load_weights(pre_weights_path)
    feature.trainable = False
    
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.3),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(rate=0.3),
                                 tf.keras.layers.Dense(10),
                                 tf.keras.layers.Softmax()])
    
    '''  
    #mobilenetv3
    model = mobilenet_v3_small(input_shape=(im_height, im_width, 3),
                               num_classes=10,
                               include_top=True)
                              
    '''
    #权重赋值1
    pre_weights_path = './weights_mobilenet_v3_small_224_1.0_float.h5'
    assert len(glob.glob(pre_weights_path + "*")), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    if freeze_layer is True:
        # freeze layer, only training 2 last layers
        for layer in model.layers:
            if layer.name not in ["Conv_2", "Logits/Conv2d_1c_1x1"]:
                layer.trainable = False
            else:
                print("training: " + layer.name)    
    
    #shuffleNet
    model = shufflenet_v2(num_classes=10,
                          input_shape=(im_height, im_width, 3),
                          stages_repeats=[4, 8, 4],
                          stages_out_channels=[24, 116, 232, 464, 1024])
    
    #efficientnet
    model = efficient_net(width_coefficient=1.0,
                          depth_coefficient=1.1,
                          input_shape=(224, 224, 3),
                          dropout_rate=0.2,
                          drop_connect_rate=0.2,
                          activation="swish",
                         model_name="efficientnet",
                          include_top=True,
                          num_classes=10)
    
    #vgg16                      
    model = vgg(model_name="vgg16", im_height=224, im_width=224, num_classes=10)
    
    #goolenet
    model = GoogLeNet(im_height=224, im_width=224, class_num=10, aux_logits=False)  
    ''' 
    # 创建特征提取模型以获取 SE 模块的输出和注意力权重
    #feature_extraction_model = create_feature_extraction_model(model)
    
    model.summary()
    

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    # 初始化准确率列表
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    # 添加其他指标列表
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []
    val_roc_auc_list = []

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)
        
    @tf.function
    def test_step(images, labels):
        output = model(images, training=False)
        t_loss = loss_object(labels, output)

        test_loss(t_loss)
        test_accuracy(labels, output)

    best_test_loss = float('inf')
    train_step_num = train_num // batch_size
    val_step_num = val_num // batch_size
    

    for epoch in range(1, epochs+1):

        train_loss.reset_states()        # clear history info
        train_accuracy.reset_states()    # clear history info
        test_loss.reset_states()         # clear history info
        test_accuracy.reset_states()     # clear history info
        
        t1 = time.perf_counter()
        for index, (images, labels) in enumerate(train_dataset):
            train_step(images, labels)
            if index+1 == train_step_num:
                break
        print(time.perf_counter()-t1)
        
        val_all_pred = []  # 初始化为一个空列表，用于收集真实标签
        val_all_labels = []  # 初始化为一个空列表，用于收集预测结果
        for index, (images, labels) in enumerate(val_dataset):
            test_step(images, labels)
            #print("Length of labels:", len(labels))

            batch_pred = model(images, training=False)  # 获取模型的预测结果
            val_all_pred.extend(batch_pred.numpy())  # 将预测结果添加到列表中
            val_all_labels.extend(labels.numpy())  # 将标签添加到列表中
            
            if index+1 == val_step_num:
                break

        # 应用 softmax 并计算评估指标
        val_pred = np.array(val_all_pred)
        val_true = np.array(val_all_labels)
        val_pred_softmax = tf.nn.softmax(val_pred)
        val_pred_labels = np.argmax(val_pred_softmax, axis=1)
        val_true = np.argmax(val_true, axis=1)
        
        # 计算Precision, Recall和F1 Score
        precision = precision_score(val_true, val_pred_labels, average='macro', zero_division=1)
        recall = recall_score(val_true, val_pred_labels, average='macro')
        f1 = f1_score(val_true, val_pred_labels, average='macro')
        
        # 计算ROC AUC，这里以二进制分类为例，多分类需要调整
        roc_auc = roc_auc_score(val_true, val_pred_softmax, multi_class='ovo')

        # 更新指标列表
        val_precision_list.append(precision)
        val_recall_list.append(recall)
        val_f1_list.append(f1)
        val_roc_auc_list.append(roc_auc)

        # 清除临时存储的预测结果和真实标签
        #val_pred = np.array([])  
        #val_true = np.array([]) 

        
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        if test_loss.result() < best_test_loss:
            model.save_weights("./save_weights/Fashion-MNIST_xiaorong.ckpt", save_format='tf')

        train_acc_list.append(train_accuracy.result() * 100)
        val_acc_list.append(test_accuracy.result() * 100)
        train_loss_list.append(train_loss.result())
        val_loss_list.append(test_loss.result())

    
    epochs = range(1, len(train_acc_list) + 1) 
    '''
    # 使用plot函数分别绘制训练准确率和验证准确率

    plt.figure(figsize=(12, 8))

    # 使用subplot创建两个y轴
    ax1 = plt.subplot(2, 1, 1)  # 第一个子图
    ax2 = plt.subplot(2, 1, 2)  # 第二个子图与第一个共享y轴


    ax1.plot(epochs, train_acc_list, 'r-', label='Training Accuracy')  # 'r-' 表示红色的实线
    ax1.plot(epochs, val_acc_list, 'b-', label='Validation Accuracy')  # 'b-' 表示蓝色的实线

    # 添加图例至右上角
    ax1.legend(loc='upper right')

    # 添加标题和轴标签
    ax1.set_title('Training and Validation Accuracy over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')

    # 显示网格
    ax1.grid(True)

    # 显示图表
    #plt.show()


    # 使用plot函数分别绘制训练损失和验证损失
    ax2.plot(epochs, train_loss_list, 'r--', label='Training Loss')  # 'r--' 表示红色的虚线
    ax2.plot(epochs, val_loss_list, 'b--', label='Validation Loss')  # 'b--' 表示蓝色的虚线

    # 添加图例至右上角
    ax2.legend(loc='upper right')

    # 添加标题和轴标签
    ax2.set_title('Training and Validation Loss over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    # 如果需要，可以设置y轴的范围
    #plt.ylim(0, 1.2)  # 例如，设置y轴的范围从0到1.2

    # 显示网格
    ax2.grid(True)
    # 调整子图间距
    plt.tight_layout()

    plt.savefig('my_figure_MobileSmall.png')  # 保存为PNG格式

    # 显示图表
    plt.show()
'''

    # 训练结束后，使用以下代码绘制所有指标的曲线
    # 将准确率列表转换为CSV格式并保存
    accuracy_csv_path = 'save_folder/Fashion-MNIST_xiaorong_accuracy.csv'  # 修改文件路径以保存到特定文件夹
    loss_csv_path = 'save_folder/Fashion-MNIST_xiaorong_loss.csv'  # 修改文件路径以保存到特定文件夹

    # 确保保存文件的文件夹存在
    os.makedirs('save_folder', exist_ok=True)

    with open(accuracy_csv_path, 'w', newline='') as file_acc, open(loss_csv_path, 'w', newline='') as file_loss:
        writer_acc = csv.writer(file_acc)
        writer_loss = csv.writer(file_loss)

        # 将 epochs、train_acc_list 和 val_acc_list 打包成元组，然后写入 CSV 文件
        writer_acc.writerows(zip(epochs, train_acc_list, val_acc_list))

        # 将 epochs、train_loss_list 和 val_loss_list 打包成元组，然后写入另一个 CSV 文件
        writer_loss.writerows(zip(epochs, train_loss_list, val_loss_list))

        print(f'Accuracy data saved to {accuracy_csv_path}')
        print(f'Loss data saved to {loss_csv_path}')
    
    plt.figure(figsize=(20, 10))

    # 计算所有epoch的平均指标
    average_train_accuracy = np.mean(train_acc_list)
    average_val_accuracy = np.mean(val_acc_list)
    average_val_precision = np.mean(val_precision_list)
    average_val_recall = np.mean(val_recall_list)
    average_val_f1 = np.mean(val_f1_list)
    average_val_roc_auc = np.mean(val_roc_auc_list)
    min_val_loss = min(val_loss_list)
    last_epoch_train_loss = train_loss_list[-1]
    last_epoch_val_loss = val_loss_list[-1]

    # 绘制训练准确率和验证准确率曲线
    ax1 = plt.subplot(231)
    ax1.plot(epochs, train_acc_list, 'r-', label='Training Accuracy')
    ax1.plot(epochs, val_acc_list, 'b-', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='center right')
    # 添加平均准确率的文本标注
    ax1.text(0.05, 0.95, 'Avg Train Accuracy: {:.2f}%'.format(average_train_accuracy), transform=ax1.transAxes, color='r')
    ax1.text(0.05, 0.90, 'Avg Val Accuracy: {:.2f}%'.format(average_val_accuracy), transform=ax1.transAxes, color='b')
    
    # 绘制训练损失和验证损失曲线
    ax2 = plt.subplot(232)
    ax2.plot(epochs, train_loss_list, 'r--', label='Training Loss')
    ax2.plot(epochs, val_loss_list, 'b--', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='center right')
    ax2.text(0.05, 0.95, 'Last Train Loss: {:.4f}\nLast Val Loss: {:.4f}'.format(last_epoch_train_loss, last_epoch_val_loss), transform=ax2.transAxes)

    # 绘制精确度曲线
    ax3 = plt.subplot(233)
    ax3.plot(epochs, val_precision_list, 'g-', label='Validation Precision')
    ax3.set_title('Validation Precision')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Precision')
    ax3.legend(loc='center right')
    ax3.text(0.05, 0.95, 'Avg Precision: {:.2f}%'.format(average_val_precision), transform=ax3.transAxes)
    # 绘制召回率曲线
    ax4 = plt.subplot(234)
    ax4.plot(epochs, val_recall_list, 'y-', label='Validation Recall')
    ax4.set_title('Validation Recall')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Recall')
    ax4.legend(loc='center right')
    ax4.text(0.05, 0.95, 'Avg Recall: {:.2f}%'.format(average_val_recall), transform=ax4.transAxes)
    # 绘制F1分数曲线
    ax5 = plt.subplot(235)
    ax5.plot(epochs, val_f1_list, 'm-', label='Validation F1 Score')
    ax5.set_title('Validation F1 Score')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('F1 Score')
    ax5.legend(loc='center right')
    ax5.text(0.05, 0.95, 'Avg F1 Score: {:.2f}%'.format(average_val_f1),    transform=ax5.transAxes)
    # 绘制ROC AUC曲线
    ax6 = plt.subplot(236)
    ax6.plot(epochs, val_roc_auc_list, 'c-', label='Validation ROC AUC')
    ax6.set_title('Validation ROC AUC')
    ax6.set_xlabel('Epochs')
    ax6.set_ylabel('ROC AUC')
    ax6.legend(loc='center right')
    ax6.text(0.05, 0.95, 'Avg ROC AUC: {:.2f}'.format(average_val_roc_auc), transform=ax6.transAxes)
    # 调整子图间距
    plt.tight_layout(pad=3.0)
    plt.savefig('Fashion-MNIST_xiaorong_plots.png') 
    plt.show()

    # 绘制混淆矩阵
    print(len(val_true))
    print(len(val_pred_labels))
    conf_matrix = confusion_matrix(val_true, val_pred_labels)
    plt.figure(figsize=(8, 6))
    annot = np.vectorize(lambda x: "{:.2f}".format(x) if x != 0 else "")(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues') 
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('Fashion-MNIST_xiaorong_matrix.png') 
    plt.show()
    


if __name__ == '__main__':
    main()


