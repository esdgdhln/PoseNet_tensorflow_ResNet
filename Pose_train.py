# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from ResNet_work import *
import cv2
from tqdm import tqdm
import time
import os
from multiprocessing import Process, Queue, Pool


batch_size = 50
# Set this path to your dataset directory
directory = '/home/sky/PoseNet-master/posenet/KingsCollege/'
dataset = 'dataset_train.txt'

class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = (new_height - output_side_length) / 2
	width_offset = (new_width - output_side_length) / 2
	cropped_img = img[height_offset:height_offset + output_side_length,
						width_offset:width_offset + output_side_length]
	return cropped_img

def preprocess(images):
	images_out = [] #final result
	#Resize and crop and compute mean!
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		X = cv2.resize(X, (455, 256))
		X = centeredCrop(X, 224)
		images_cropped.append(X)
	#compute images mean
	N = 0
	mean = np.zeros((1, 3, 224, 224))
	for X in tqdm(images_cropped):
		mean[0][0] += X[:,:,0]
		mean[0][1] += X[:,:,1]
		mean[0][2] += X[:,:,2]
		N += 1
	mean[0] /= N
	#Subtract mean from all images
	for X in tqdm(images_cropped):
		X = np.transpose(X,(2,0,1))
		X = X - mean
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		images_out.append(X)
	return images_out

def get_data():
	poses = []
	images = []

	with open(directory+dataset) as f:
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			p4 = float(p4)
			p5 = float(p5)
			p6 = float(p6)
			poses.append((p0,p1,p2,p3,p4,p5,p6))
			images.append(directory+fname)
	images = preprocess(images)
	return datasource(images, poses)

def gen_data(source):
	while True:
		indices = range(len(source.images))
		random.shuffle(indices)
		for i in indices:
			image = source.images[i]
			pose_x = source.poses[i][0:3]
			pose_q = source.poses[i][3:7]
			yield image, pose_x, pose_q

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)


class Train(object):
    def __init__(self):
        self.placeholders()
    def placeholders(self):
        self.images_placeholder = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
        self.poses_x_placeholder = tf.placeholder(tf.float32, [batch_size, 3])
        self.poses_q_placeholder = tf.placeholder(tf.float32, [batch_size, 4])

    def build_train_graph(self):
        global_step = tf.Variable(0, trainable=False)
        pose = inference(self.images_placeholder, FLAGS.num_residual_blocks, reuse=False)
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(pose, self.poses_x_placeholder,self.poses_q_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)
        self.train_op = self.train_operation(global_step, self.full_loss)


    def train_operation(self,global_step,total_loss):
        opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001,
                                     use_locking=False, name='Adam').minimize(self.full_loss,global_step=global_step)
        return opt


    def loss(self,logits,translation,rotation):
        #lx,lq = logits[0:3],logits[3:7]
        lx = tf.slice(logits, [0, 0], [FLAGS.batch_size, 3])
        lq = tf.slice(logits, [0, 3], [FLAGS.batch_size, 4])
        #lx, lq = logits[0:3], logits[3:7]
        l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(lx, translation)))) * 0.3
        l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(lq, rotation)))) * 150
        cost = l1_x +l1_q
        return cost

    def train(self):
        datasource = get_data()
        self.build_train_graph()
        saver = tf.train.Saver(tf.global_variables())
        init = tf.initialize_all_variables()


        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.880) config=tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session() as sess:
            if FLAGS.is_use_ckpt is True:
                saver.restore(sess,FLAGS.ckpt_path)
                print 'restored from checkpoints'

            else:
                sess.run(init)
            step_list = []
            train_error_list = []
            print 'Start training...'
            data_gen = gen_data_batch(datasource)
            for step in xrange(FLAGS.train_steps):
                np_images, np_poses_x, np_poses_q = next(data_gen)
                feed = {self.images_placeholder: np_images, self.poses_x_placeholder: np_poses_x, self.poses_q_placeholder: np_poses_q}
                #start_time = time.time()
                _,train_loss = sess.run([self.train_op,self.full_loss],feed_dict=feed)
                #duration = time.time() - start_time()

                if step % 500 == 0 or (step + 1) == FLAGS.train_steps:
                    print('train loss', train_loss)
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)



    
    
if __name__ == '__main__':
    train = Train()
    train.train()


