# Slightly modified the code from "PhysicsGuidedNeuralNetworksforSpatio-temporalSuper-resolutionof TurbulentFlows" by Bao et al. (2022)

from __future__ import print_function, division

from tensorflow.keras.layers import *
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import datetime
#from data_loader import DataLoader
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from extra import generate_image

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# a = np.load('/content/drive/My Drive/PGSRN(Model and Data)/PGSRN_across_time/data90/u_les_all.npy')
# b = np.load('/content/drive/My Drive/PGSRN(Model and Data)/PGSRN_across_time/data90/v_les_all.npy')
# c = np.load('/content/drive/My Drive/PGSRN(Model and Data)/PGSRN_across_time/data90/w_les_all.npy')

# Generating data

a = np.ones(shape=(2600, 32, 32))
c = np.zeros(shape=(2600, 32, 32))

# generate 32x32 images of fluid flow
b = generate_image(2600, 32)

print(a.shape)
print(b.shape)
print(c.shape)
maximum0 = a.max()
minimum0 = a.min()
print(maximum0,minimum0)
maximum1 = b.max()
minimum1 = b.min()
print(maximum1,minimum1)
maximum2 = c.max()
minimum2 = c.min()
print(maximum2,minimum2)

dataset = np.zeros((2600,32,32,3))


for i in range(2600):
	temp = a[i]#u
	temp1 = b[i]#v
	temp2 = c[i]#w
	#u
	#temp = (temp - minimum0)/(maximum0 - minimum0)
	#v
	#temp1 = (temp1 - minimum1)/(maximum1 - minimum1)
	#w
	#temp2 = (temp2 - minimum2)/(maximum2 - minimum2)
	

	for j in range(32):
		for k in range(32):
			dataset[i][j][k] = np.array([temp[j][k],temp1[j][k],temp2[j][k]])
print(dataset.shape)
#print(dataset)
print(dataset.min())
print(dataset.max())

######################DNS##################################################



######################DNS##################################################
# read in dns data from npy file
# a = np.load('/content/drive/My Drive/PGSRN(Model and Data)/PGSRN_across_time/data90/u_dns_all.npy')
# b = np.load('/content/drive/My Drive/PGSRN(Model and Data)/PGSRN_across_time/data90/v_dns_all.npy')
# c = np.load('/content/drive/My Drive/PGSRN(Model and Data)/PGSRN_across_time/data90/w_dns_all.npy')

# Generating data

a = np.ones(shape=(2600, 128, 128))
c = np.zeros(shape=(2600, 128, 128))

# generate 128x128 images of fluid flow
b = generate_image(2600, 128)


print(a.shape)
print(b.shape)
print(c.shape)
dataset1 = np.zeros((2600,128,128,3))

maximum0_dns = a.max()
minimum0_dns = a.min()
print(maximum0_dns,minimum0_dns)
maximum1_dns = b.max()
minimum1_dns = b.min()
print(maximum1_dns,minimum1_dns)
maximum2_dns = c.max()
minimum2_dns = c.min()
print(maximum2_dns,minimum2_dns)


for i in range(2600):
	temp = a[i]#u
	temp1 = b[i]#v
	temp2 = c[i]#w
	#u
	#temp = (temp - minimum0_dns)/(maximum0_dns - minimum0_dns)
	#v		
	#temp1 = (temp1 - minimum1_dns)/(maximum1_dns - minimum1_dns)
	#w
	#temp2 = (temp2 - minimum2_dns)/(maximum2_dns - minimum2_dns)

		
	for j in range(128):
		for k in range(128):
			dataset1[i][j][k] = np.array([temp[j][k],temp1[j][k],temp2[j][k]])

print(dataset1.shape)
print(dataset1.min())
print(dataset1.max())



def mean(vects):
	u_all = vects[:,:,:,0]
	v_all = vects[:,:,:,1]
	w_all = vects[:,:,:,2]
				
	u = u_all[:65,:,:]
	v = v_all[:65,:,:]
	w = w_all[:65,:,:]

	
	u = u - tf.reduce_mean(u)   
	v = v - tf.reduce_mean(v) 
	w = w - tf.reduce_mean(w) 

	vects = tf.stack([u,v,w],axis=3)

	return vects




####build the model
class SRGAN():
	def __init__(self):
		# Input shape
		self.channels = 3
		self.lr_height = 32  # Low resolution height
		self.lr_width = 32  # Low resolution width
		self.lr_shape = (self.lr_height, self.lr_width, self.channels)
		self.hr_height = 128  # High resolution height
		self.hr_width = 128  # High resolution width
		self.hr_shape = (self.hr_height, self.hr_width, self.channels)

		# Number of residual blocks in the generator
		self.n_residual_blocks = 16

		optimizer = Adam(0.0002, 0.5)

		# We use a pre-trained VGG19 model to extract image features from the high resolution
		# and the generated high resolution images and minimize the mse between them
		self.vgg = self.build_vgg()
		self.vgg.trainable = False
		self.vgg.summary()
		print("vgg model summary")
		print(self.vgg.summary())
		
		self.vgg.compile(loss='mse',
						 optimizer=optimizer,
						 metrics=['mean_squared_error'])


		# Calculate output shape of D (PatchGAN)
		patch = int(self.hr_height / 2 ** 4)
		self.disc_patch = (patch, patch, 1)

		# Number of filters in the first layer of G and D
		self.gf = 64
		self.df = 64

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		print("discriminator model")
		print(self.discriminator.summary())
		self.discriminator.compile(loss=["mse","mse"],
								   optimizer=optimizer,
								   metrics=['mean_squared_error'])


		# Build the generator
		self.generator= self.build_generator()
		print("generator model")
		print(self.generator.summary())

		# High res. and low res. images
		img_hr = Input(shape=self.hr_shape)
		img_lr = Input(shape=self.lr_shape)


		# Generate high res. version from low res.
		fake_hr,fake_hrm = self.generator(img_lr)
		

		# Extract image features of the generated img
		fake_features = self.vgg(fake_hr)
		fake_features_m = self.vgg(fake_hrm)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# Discriminator determines validity of generated high res. images
		validity, fake_les = self.discriminator(fake_hr)
#		
		self.combined = Model([img_lr,img_hr], [validity, fake_features,fake_hr])

		self.combined.compile(loss=['binary_crossentropy', "mse", self.divergent_loss_tr],
							  loss_weights=[1e-3, 1, 100], 
							  optimizer=optimizer,metrics=['mean_squared_error'])	
	#self.binary_crossentropy					
	def mean_squared_error(self, y_true, y_pred):
		yt = tf.reshape(y_true[:,:,:,:],[-1])
		yp = tf.reshape(y_pred[:,:,:,:],[-1])
		return  tf.reduce_mean(tf.square(yt-yp))
#			
					
			
	def divergent_loss_tr(self, y_true, y_pred):
		u = y_pred[:,:,:,0]
		v = y_pred[:,:,:,1]
		w = y_pred[:,:,:,2]
		

		img_size = 128	
		num_img = 65
		u1 = tf.concat([tf.zeros([num_img,1,img_size]),u[:,2:,:],tf.zeros([num_img,1,img_size])],axis=1)
		u2 = tf.concat([tf.zeros([num_img,1,img_size]),u[:,:-2,:],tf.zeros([num_img,1,img_size])],axis=1)
		du = (u1-u2)
		#	print(du.shape)
			
		v1 = tf.concat([tf.zeros([num_img,img_size,1]),v[:,:,2:],tf.zeros([num_img,img_size,1])],axis=2)
		v2 = tf.concat([tf.zeros([num_img,img_size,1]),v[:,:,:-2],tf.zeros([num_img,img_size,1])],axis=2)
		dv = (v1-v2)
		#	print(dv.shape)
			
		w1 = tf.concat([tf.zeros([1,img_size,img_size]),w[2:,:,:],tf.zeros([1,img_size,img_size])],axis=0)
		w2 = tf.concat([tf.zeros([1,img_size,img_size]),w[:-2,:,:],tf.zeros([1,img_size,img_size])],axis=0)
		dw = (w1-w2)

		s = du+dv+dw/2
		s = tf.reshape(s,[-1])

		cost_phy = tf.reduce_mean(tf.square(s))
		
		return cost_phy

	def physical_loss_tr(self, y_true, y_pred):
		u = y_pred[:,:,:,0]
		v = y_pred[:,:,:,1]
		w = y_pred[:,:,:,2]

		u = tf.reshape(u,[-1])
		v = tf.reshape(v,[-1])
		w = tf.reshape(w,[-1])
		
		cost_phy = tf.reduce_mean(tf.square(u)) + tf.reduce_mean(tf.square(v)) + tf.reduce_mean(tf.square(w))
		
		return cost_phy


		
	def build_vgg(self):
		"""
		Builds a pre-trained VGG19 model that outputs image features extracted at the
		third block of the model
		"""
		print('start loading trianed weights of vgg...')
		vgg = VGG19(weights="imagenet", include_top=False)
		# Set outputs to outputs of last conv. layer in block 3
		# See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
		print('loading completes')
		vgg.outputs = [vgg.layers[9].output]

		img = Input(shape=self.hr_shape)

		# Extract image features
		img_features = vgg(img)

		return Model(img, img_features)

	def build_generator(self):

		def residual_block(layer_input, filters):
			"""Residual block described in paper"""
			d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
			d = Activation('relu')(d)
			d = BatchNormalization(momentum=0.8)(d)
			d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
			d = BatchNormalization(momentum=0.8)(d)
			d = Add()([d, layer_input])
			return d

		def deconv2d(layer_input):
			"""Layers used during upsampling"""
			u = UpSampling2D(size=2)(layer_input)
			u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
			u = Activation('relu')(u)
			return u

		# Low resolution image input
		img_lr = Input(shape=self.lr_shape)

		# Pre-residual block
		c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
		c1 = Activation('relu')(c1)

		# Propogate through residual blocks
		r = residual_block(c1, self.gf)
		for _ in range(self.n_residual_blocks - 1):
			r = residual_block(r, self.gf)

		# Post-residual block
		c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
		c2 = BatchNormalization(momentum=0.8)(c2)
		c2 = Add()([c2, c1])
		gen_hr_m1 = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(c2)

		# Upsampling
		u1 = deconv2d(c2)
		gen_hr_m = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u1)
		u2 = deconv2d(u1)

		# Generate high resolution output
		gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)
		out = Lambda(mean, output_shape=self.hr_shape)(gen_hr)

		#return Model(img_lr, [gen_hr, out])
		return Model(img_lr, [out, gen_hr])
	def build_discriminator(self):

		def d_block(layer_input, filters, strides=1, bn=True):
			"""Discriminator layer"""
			d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			return d

		# Input img
		d0 = Input(shape=self.hr_shape)

		d1 = d_block(d0, self.df, bn=False)
		d2 = d_block(d1, self.df, strides=2)
		d3 = d_block(d2, self.df * 2)
		d4 = d_block(d3, self.df * 2, strides=2)
		d5 = d_block(d4, self.df * 4)
		out = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(d5)
		
		d6 = d_block(d5, self.df * 4, strides=2)
		d7 = d_block(d6, self.df * 8)
		d8 = d_block(d7, self.df * 8, strides=2)

		d9 = Dense(self.df * 16)(d8)
		d10 = LeakyReLU(alpha=0.2)(d9)
		validity = Dense(1, activation='sigmoid')(d10)

		return Model(d0, [validity, out])
		

	def train(self, epochs, batch_size, No, LES,DNS):

		
		rmse_output = []
		start_time = datetime.datetime.now()	
		ru = 10 
		rv = 10 
		rw = 10 
		for epoch in range(1,epochs+1):

			
			# ----------------------
			#  Train Discriminator
			# ----------------------

			# Sample images and their conditioning counterparts
			print("###############################################################r")			
			print("training discriminator")
			for i in range(20):
				
				print("epoch: " + str(epoch)+", batch: "+str(i))
				print("training no: ", 0+65*i,65+65*i)

				
				imgs_hr = DNS[0+65*i:65+65*i,:,:,:]
				imgs_lr = LES[0+65*i:65+65*i,:,:,:]
#				imgs_hr, imgs_lr = DNS[0+65*i:65+65*i,:,:,:], LES[0+65*i:65+65*i,:,:,:]
		
				# From low res. image generate high res. version
				fake_hr,fake_hrm = self.generator.predict(imgs_lr)

				
				#calculate mean square error:
				
				valid = np.ones((batch_size,) + self.disc_patch)
				fake = np.zeros((batch_size,) + self.disc_patch)
				
					
				# Train the discriminators (original images = real / generated = Fake)
				d_loss_real = self.discriminator.train_on_batch(imgs_hr, [valid,imgs_lr])
				d_loss_fake = self.discriminator.train_on_batch(fake_hr, [fake,imgs_lr])
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
				print(epoch,'d',"batch: "+str(i),d_loss)

			# ------------------
			#  Train Generator
			# ------------------
			print("training generator")
			for i in range(20):
				print("epoch: " + str(epoch)+", batch: "+str(i))
				print("training no: ", 0+65*i,65+65*i)


				# Sample images and their conditioning counterparts
				imgs_hr = DNS[0+65*i:65+65*i,:,:,:]
				imgs_lr = LES[0+65*i:65+65*i,:,:,:]
				


				# The generators want the discriminators to label the generated images as real
				valid = np.ones((batch_size,) + self.disc_patch)

				# Extract ground truth image features using pre-trained VGG19 model
				image_features = self.vgg.predict(imgs_hr)


				# Train the generators
				g_loss = self.combined.train_on_batch([imgs_lr,imgs_hr], [valid, image_features, imgs_hr])
				print(epoch, "g","batch "+str(i),g_loss)
#				print(type(g_loss))

			elapsed_time = datetime.datetime.now() - start_time
			# Plot the progress
			print("%d time: %s" % (epoch, elapsed_time))
			
			# If at save interval => save generated image samples
			if epoch % 10 == 0 and epoch > 100 and No == 1:
				self.discriminator.save_weights('/content/drive/My Drive/PGSRN(Model and Data)/PGSRN_across_time1130/model_D/' + str(epoch) + 'save_0_500.h5')
				self.generator.save_weights('/content/drive/My Drive/PGSRN(Model and Data)/PGSRN_across_time1130/model_G/' + str(epoch) + 'save_0_500.h5')
			
			print("calculate all testing RMSE: ")
			imgs_hr, imgs_lr = DNS,LES
			imgs_hr_test, imgs_lr_test = DNS[1300:1625,:,:,:],LES[1300:1625,:,:,:]
			
			
			p1,fake_hrm1_ = gan.generator.predict(imgs_lr_test[:65,:,:,:])
			p2,fake_hrm2 = gan.generator.predict(imgs_lr_test[65:130,:,:,:])
			p3,fake_hrm3 = gan.generator.predict(imgs_lr_test[130:195,:,:,:])
			p4,fake_hrm4 = gan.generator.predict(imgs_lr_test[195:260,:,:,:])
			p5,fake_hrm5 = gan.generator.predict(imgs_lr_test[260:,:,:,:])
			print(p1.shape, p2.shape, p3.shape, p4.shape, p5.shape)
			
			pred = np.concatenate((p1,p2,p3,p4,p5))
			print(pred.shape)
			a = pred[:,:,:,0]#*(maximum0_dns - minimum0_dns)+ minimum0_dns#u
			b = pred[:,:,:,1]#*(maximum1_dns - minimum1_dns)+ minimum1_dns#v
			c = pred[:,:,:,2]#*(maximum2_dns - minimum2_dns)+ minimum2_dns#w
			
            
			a1 = imgs_hr_test[:,:,:,0]#*(maximum0_dns - minimum0_dns)+ minimum0_dns#u
			b1 = imgs_hr_test[:,:,:,1]#*(maximum1_dns - minimum1_dns)+ minimum1_dns#v
			c1 = imgs_hr_test[:,:,:,2]#*(maximum2_dns - minimum2_dns)+ minimum2_dns#w
			
			rmse_ua = np.sqrt(np.sum(np.square(a[:,:,:]-a1[:,:,:]))/(325*128*128))
			rmse_va = np.sqrt(np.sum(np.square(b[:,:,:]-b1[:,:,:]))/(325*128*128))
			rmse_wa = np.sqrt(np.sum(np.square(c[:,:,:]-c1[:,:,:]))/(325*128*128))

			print("RMSE testing all:",rmse_ua,rmse_va,rmse_wa)


######for training
gan = SRGAN()
gan.train(500, 65,1, dataset,dataset1)