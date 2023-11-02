import tensorflow as tf
import scipy.io
import numpy as np
from ops_ExprGAN_P3 import *
from scipy.io import loadmat
import time
from time import gmtime, strftime
from tensorflow.python.framework import ops 
np.random.seed(2019)

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

class SRVF_WGAN(object):
    def __init__(self,
                 session,
                 size_SRVF_H = 204,
                 size_SRVF_W = 30,
                 size_kernel=5,
                 size_batch=128,
                 num_encoder_channels=64,
                 num_z_channels=50,
                 num_input_channels =1,
                 y_dim=12,
                 rb_dim=3,
                 num_gen_channels=1024,
                 enable_tile_label=False,
                 tile_ratio=1.0,
                 is_training=True,
                 disc_iters = 4, # For WGAN and WGAN-GP, number of descri iters per gener iter
                 is_flip=True,
                 discription='wLoss10_LR_6_geoloss',    ###'wLoss100',   ###"WithDecayLearingRate1000",
                 checkpoint_dir='./checkpoint',
                 save_dir='Results/',
                 num_epochs=600,
                 learning_rate=0.000001,
                 LAMBDA = 10, # Gradient penalty lambda hyperparameter
                 param_help=0,
                 w_loss=10
                 ):
         
        self.session = session
        self.param_help=param_help
        self.size_SRVF_H = size_SRVF_H
        self.size_SRVF_W = size_SRVF_W
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels=num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.y_dim = y_dim
        self.rb_dim = rb_dim
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.is_flip = is_flip
        self.checkpoint_dir = checkpoint_dir + discription
        self.disc_iters=disc_iters
        self.num_epochs=num_epochs
        self.learning_rate=learning_rate  
        self.LAMBDA = LAMBDA  
        self.discription= discription
        self.w_loss= w_loss
        print(self.discription)
  
        ###reference point
        Q_ref_=loadmat('../Data/FaceTalk_170913_03279_TA_mouth_up_1_SRVF.mat')
        #print(Q_ref_)
        q_mean=Q_ref_["q2n"]
        self.Q_ref=np.zeros([self.size_batch,  self.size_SRVF_H, self.size_SRVF_W])
        for i in range(self.Q_ref.shape[0]):
            self.Q_ref[i,:,:]=q_mean

        self.Q_ref_tensor=tf.constant(self.Q_ref, dtype=tf.float32)
    
        print("\n\tLoading data")
        self.data_X, self.data_y = self.load_data('../Data/COMA_SRVF_neutral2Exp')
        self.data_X=[os.path.join("../Data/COMA_SRVF_neutral2Exp", x) for x in self.data_X]

        #input_data=    tf.placeholder(
        #            tf.float32,
        #            [self.size_batch, self.size_SRVF_H, self.size_SRVF_W, self.num_input_channels],
        #            name='input_data'
        #        )      
        self.real_data = tf.compat.v1.placeholder(
            tf.float32,
            [self.size_batch,  self.size_SRVF_H*self.size_SRVF_W],
            name='real_data'
        )
        self.emotion = tf.compat.v1.placeholder(
            tf.float32,
            [self.size_batch, self.y_dim*self.rb_dim],
            name='emotion_labels'
        )




        self.log_real=self.log_map(self.real_data)
        self.fake_data = self.Generator(self.emotion)
        self.exp_fake=self.exp_map(self.fake_data)
        self.log_exp_fake=self.log_map(self.exp_fake)
        
        self.disc_log_real = self.Discriminator(self.log_real, self.emotion, enable_bn=True)
        self.disc_log_exp_fake = self.Discriminator(self.log_exp_fake, self.emotion, reuse_variables=True, enable_bn=True)
        ############### losses to minimize (needs exp_map and log_map gradients !!!!)
        ################ TODO: try L2 loss for reconstruction 
        ## reconstruction_loss = tf.nn.l2.loss(log_exp_fake - log_real)   #L2 loss
        reconstruction_loss = tf.reduce_mean(tf.abs(self.log_real - self.log_exp_fake))# L1 loss
        geodesic_reconstruction_loss= tf.reduce_mean(self.geodesic_dist(self.exp_fake, self.real_data))
        self.gen_cost = -tf.reduce_mean(self.disc_log_exp_fake) + self.w_loss*reconstruction_loss + self.w_loss*geodesic_reconstruction_loss
        self.gen_cost_= -self.gen_cost
        self.help_loss = tf.reduce_mean(self.disc_log_real)
        self.disc_cost = tf.reduce_mean(self.disc_log_exp_fake) - tf.reduce_mean(self.disc_log_real) ###+ reconstruction_loss


        # penalty of improved WGAN
        alpha = tf.compat.v1.random_uniform(
           shape=[self.size_batch,1], 
           minval=0.,
           maxval=1.
           )
        differences = self.log_exp_fake - self.log_real
        interpolates = self.log_real + (alpha*differences)   
        gradients = tf.gradients(self.Discriminator(interpolates, self.emotion, reuse_variables=True, enable_bn=True), [interpolates])[0]
        ##slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        slopes = tf.sqrt(tf.compat.v1.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        self.disc_cost += self.LAMBDA*gradient_penalty

        trainable_variables = tf.compat.v1.trainable_variables() ##returns all variables created(the two variable scopes) and makes trainable true
        self.gen_params = [var for var in trainable_variables if 'G_' in var.name]
        self.disc_params = [var for var in trainable_variables if 'D_' in var.name]


        GEN_cost_summary = tf.compat.v1.summary.scalar('GEN_cost', self.gen_cost_)   ##scalar_summary('GEN_cost', self.gen_cost_)
        DISC_cost_summary = tf.compat.v1.summary.scalar('DISC_cost',self. disc_cost)
        reconstruction_cost_summary = tf.compat.v1.summary.scalar('Reconstruction_cost', reconstruction_loss)
        #geodesic_reconstruction_loss_summary = tf.summary.scalar('Geodesic_reconstruction_cost', geodesic_reconstruction_loss)
        help_cost_summary = tf.compat.v1.summary.scalar('DiscReal_cost', self.help_loss)
        #self.summary = tf.summary.merge([GEN_cost_summary, DISC_cost_summary, geodesic_reconstruction_loss_summary, reconstruction_cost_summary, help_cost_summary])  #merge_summary
        self.summary = tf.compat.v1.compat.v1.summary.merge([GEN_cost_summary, DISC_cost_summary, reconstruction_cost_summary,help_cost_summary])  # merge_summary

        #self.saver = tf.compat.v1.train.Saver(max_to_keep=10)   ##keep_checkpoint_every_n_hours=3 )  ##max_to_keep=10,
        self.saver = tf.compat.v1.train.Saver(max_to_keep = 30, keep_checkpoint_every_n_hours = 3)
  
    def train(self,
              num_epochs=200,
              learning_rate=0.0002,
              decay_rate=1.0,
              enable_shuffle=True,
              use_trained_model=True,
              ):

       
        ## count number of batches seen by the graph
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        Train_learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=1000,  ##len(self.data_X) // self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
            )
        with tf.compat.v1.variable_scope('gen-optimize', reuse=tf.compat.v1.AUTO_REUSE):  ##tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
          self.gen_train_op = tf.compat.v1.train.AdamOptimizer(
                  learning_rate=Train_learning_rate, 
                   beta1=0.5,
                   beta2=0.9
                   ).minimize(self.gen_cost, global_step=self.global_step, var_list=self.gen_params)
        with tf.compat.v1.variable_scope('disc-optimizer', reuse=tf.compat.v1.AUTO_REUSE):
          self.disc_train_op = tf.compat.v1.train.AdamOptimizer(
                  learning_rate=Train_learning_rate, 
                  beta1=0.5, 
                  beta2=0.9
                  ).minimize(self.disc_cost, global_step=self.global_step, var_list=self.disc_params)
        
        ## write summary          
        filename='summary' + str(self.learning_rate)+ self.discription
        self.writer = tf.compat.v1.summary.FileWriter(os.path.join(self.save_dir, filename), self.session.graph)  ##train.SummaryWriter

        ###self.session.run(tf.global_variables_initializer())  ###tf.initialize_all_variables()) 
        try:
          tf.global_variables_initializer().run()
        except:
          tf.compat.v1.initialize_all_variables().run()
        num_batches = len(self.data_X) // self.size_batch     
        for epoch in range(num_epochs):
            if enable_shuffle:
                seed = 2019
                np.random.seed(seed)
                np.random.shuffle(self.data_X)
                np.random.seed(seed)
                np.random.shuffle(self.data_y)
            for ind_batch in range(num_batches):
                start_time = time.time()
                batch_files = self.data_X[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch = [self.read_SRVF(
                    path_SRVF=batch_file)for batch_file in batch_files]                  
                    ##is_flip=self.is_flip
                    ##for batch_file in batch_files]
                batch_SRVF = np.array(batch).astype(np.float32)
                ##batch_SRVF = self.log_numpy_map(batch_SRVF)
                batch_label_emo = self.data_y[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch_label_rb = np.zeros(shape=(self.size_batch, self.rb_dim * self.y_dim), dtype=np.float32)
                for i in range(self.size_batch):
                    batch_label_rb[i,:] = self.y_to_rb_label(batch_label_emo[i,:]) 
                batch_label_emo=batch_label_rb
                G_err, _ = self.session.run([self.gen_cost_, self.gen_train_op], feed_dict={self.real_data: batch_SRVF, self.emotion: batch_label_emo})
                for I in range(self.disc_iters):
                    #batch_files = self.data_X[(ind_batch+I)*self.size_batch:(ind_batch+I+1)*self.size_batch]
                    #batch = [self.read_SRVF(
                    #         path_SRVF=batch_file)for batch_file in batch_files]
                    #batch_SRVF = np.array(batch).astype(np.float32)
                    #batch_SRVF = self.log_map(batch_SRVF)
                    #batch_label_emo = self.data_y[(ind_batch+I)*self.size_batch:(ind_batch+I+1)*self.size_batch]
                    D_err, _ = self.session.run([self.disc_cost, self.disc_train_op], feed_dict={self.real_data: batch_SRVF, self.emotion: batch_label_emo})
        
                print(("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tD_err=%.4f \n\tG_err=%.4f" %
                      (epoch+1, num_epochs, ind_batch+1, num_batches, D_err, G_err )))
                #print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\" % 
                #       (epoch+1, num_epochs, ind_batch+1, num_batches))
                elapse = time.time() - start_time
                time_left = ((self.num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print(("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60)))  
                summary = self.summary.eval(
                        feed_dict={
                        self.real_data: batch_SRVF,
                        self.emotion: batch_label_emo,
                    }
                )

                self.writer.add_summary(summary, self.global_step.eval())   
                if np.mod(epoch, 10) == 1:
                     self.save_checkpoint()  
        self.save_checkpoint()
        self.writer.close()
           
            
        
        
        
    def Generator(self, y, noise=None, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
######### Encoder ######
      if reuse_variables:
            tf.get_variable_scope().reuse_variables()
      if noise is None:
        noise = tf.compat.v1.random_normal([self.size_batch, 128])
        noise=tf.concat([noise, y], 1)
      if (self.size_SRVF_H > self.size_SRVF_W):
         num_layers = int(np.log2(self.size_SRVF_W)) - int(self.size_kernel / 2)
      else:
         num_layers = int(np.log2(self.size_SRVF_H)) - int(self.size_kernel / 2)

      print("this is Generator \n")
      print("number of layers ",  num_layers)    
  ## TODO: try concat_label used in ExprGAN. In this case, 6 channels will be added to the output without changing the size of feature maps
      duplicate = 1
      z = concat_label_newtf(noise, y, duplicate=duplicate)
      ##z=noise

      ##y.get_shape()
      ##noise.get_shape()
      ##z.get_shape()
      size_mini_map_H = int(self.size_SRVF_H / 2 ** num_layers)
      size_mini_map_W = int(self.size_SRVF_W / 2 ** num_layers)
      name = 'G_fc'
      current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map_H * size_mini_map_W,
            name=name
        )
      print("number of out FC1", self.num_gen_channels * size_mini_map_H * size_mini_map_W)
      print("this is the shape of output of FC1", current.get_shape())      
      current = tf.reshape(current, [-1, size_mini_map_H, size_mini_map_W, self.num_gen_channels])
      current = tf.nn.relu(current)
      current = concat_label_newtf(current, y)
      for i in range(num_layers):
            print(i)
            name = 'G_deconv' + str(i)
            current = tf.compat.v1.image.resize_nearest_neighbor(current, [size_mini_map_H * 2 ** (i + 1), size_mini_map_W * 2 ** (i + 1)])
            current = custom_conv2d(input_map=current, num_output_channels=int(self.num_gen_channels / 2 ** (i + 1)), name=name)
            current = tf.nn.relu(current)
            current = concat_label_newtf(current, y)
            print(current.get_shape())
      name = 'G_deconv' + str(i + 1)
      current = tf.compat.v1.image.resize_nearest_neighbor(current, [self.size_SRVF_H, self.size_SRVF_W])
      #print(current.get_shape())
      current = custom_conv2d(input_map=current, num_output_channels=int(self.num_gen_channels / 2 ** (i + 2)), name=name)
      #print(current.get_shape())
      current = tf.nn.relu(current)
  
      current = concat_label_newtf(current, y)
      name = 'G_deconv' + str(i + 2)
      current = custom_conv2d(input_map=current, num_output_channels=self.num_input_channels, name=name)   ### output format: NHWC
      generated_image=tf.nn.tanh(current)
      #print(generated_image.get_shape())
      return tf.reshape(generated_image, [self.size_batch,self.size_SRVF_H*self.size_SRVF_W])




    def Discriminator( self, z, y, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16), enable_bn=True):
################# ExprGAN Descriminator ##################
      print("this is discriminator")
      if reuse_variables:
            #tf.compat.v1.variable_scope().reuse_variables()
            tf.compat.v1.get_variable_scope().reuse_variables()
      num_layers = len(num_hidden_layer_channels)
      current = tf.reshape(z, [self.size_batch ,self.size_SRVF_H, self.size_SRVF_W, 1])  ##generated_image
      current = concat_label_newtf(current, y)
      for i in range(num_layers):
            print(i)
            name = 'D_img_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    name=name
                )
            print((current.get_shape()))
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.compat.v1.layers.batch_normalization(
                    current,
                    scale=False,
                    training=is_training,
                    name=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
            current = concat_label_newtf(current, y)
            print((current.get_shape()))
      name = 'D_img_fc1'
      current = fc(
          input_vector=tf.reshape(current, [self.size_batch, -1]),
          num_output_length=1024,
        name=name
        )
      name = 'D_img_fc1_bn'
      current = tf.compat.v1.layers.batch_normalization(
            current,
            scale=False,
            training=is_training,
            name=name,
            reuse=reuse_variables
        )
      current = lrelu(current)
      current = concat_label_newtf(current, y)
      name = 'D_img_fc2'
      disc = fc(
            #input_vector=shared,
            input_vector=current,
            num_output_length=1,
            name=name
        )
      return disc
  
  
  
    def load_data(self, path_) :
       X=[]
       y=[]
       for sample in os.listdir(path_):
            #data=line.split()
            X.append(sample)
            y.append(self.read_COMA_label(sample))
       seed = 2019
       np.random.seed(seed)
       np.random.shuffle(X)
       np.random.seed(seed)
       np.random.shuffle(y)
       y_vec = np.zeros(shape=(len(y), self.y_dim), dtype=np.float32)
       for i, label in enumerate(y):
            #y_vec[i, label-1] = 1
            y_vec[i, label] = 1
       return X, y_vec

    def read_COMA_label(self, char_label):
        if 'bareteeth' in char_label:
            label=0
        elif 'cheeks_in' in char_label:
            label = 1
        elif 'eyebrow' in char_label:
            label = 2
        elif 'high_smile' in char_label:
            label = 3
        elif 'lips_back' in char_label:
            label = 4
        elif 'lips_up' in char_label:
            label = 5
        elif 'mouth_down' in char_label:
            label = 6
        elif 'mouth_extreme' in char_label:
            label = 7
        elif 'mouth_middle' in char_label:
            label = 8
        elif 'mouth_open' in char_label:
            label = 9
        elif 'mouth_side' in char_label:
            label = 10
        elif 'mouth_up' in char_label:
            label = 11
        return label
  
    def read_SRVF(self, path_SRVF):
      data_= loadmat(path_SRVF)
      ##print(path_SRVF)
      data=data_['q2n']
      data=np.reshape(data, [data.shape[0]*data.shape[1]])
      return data

    def Inner(self,A,B):
      [m, n, T]=A.get_shape().as_list()
      mult=tf.map_fn(lambda a_b: a_b[0]*a_b[1], (A,B), dtype=tf.float32)   ##A*A  ##tf.multiply(A,A)
      s1=tf.reduce_sum(mult,1, keepdims=False)
      s2=tf.reduce_sum(s1,1,keepdims=False)/T
      ##norm=tf.sqrt(s2)
      return s2


    def exp_map(self, q):
      q=tf.reshape(q, [self.size_batch,  self.size_SRVF_H, self.size_SRVF_W])
      [m, n, T]=q.get_shape().as_list()
      lw=tf.sqrt(self.Inner(q, q))
      res = self.Q_ref_tensor * tf.expand_dims(tf.expand_dims(tf.cos(lw),-1),-1) + q * (tf.expand_dims(tf.expand_dims(tf.sin(lw)/lw,-1),-1))
      return tf.reshape(res, [self.size_batch,  self.size_SRVF_H*self.size_SRVF_W])


    def log_map(self, q):
      q=tf.reshape(q, [self.size_batch,  self.size_SRVF_H, self.size_SRVF_W])
      [m, n, T]=q.get_shape().as_list()
      prod = self.Inner(self.Q_ref_tensor, q)
      u = q - self.Q_ref_tensor*tf.expand_dims(tf.expand_dims(prod,-1),-1)
      u=tf.cast(u, tf.float32)
      lu = tf.sqrt(self.Inner(u,u))
      theta = tf.acos(tf.clip_by_value(prod,-0.98, 0.98))  ###tf.acos(prod)
      zero=tf.constant(0, shape=[m], dtype=tf.float32)
      def f1(): return tf.cast(u*tf.expand_dims(tf.expand_dims(zero,-1),-1), tf.float32)
      def f2(): return tf.cast(u*tf.expand_dims(tf.expand_dims(theta/lu,-1),-1),tf.float32)
      res=tf.cond(tf.reduce_all(tf.equal(lu, zero)), f1, f2)
      return tf.reshape(res, [self.size_batch,  self.size_SRVF_H*self.size_SRVF_W])


    
    def geodesic_dist(self, q1,q2):
        q1=tf.reshape(q1, [self.size_batch,  self.size_SRVF_H, self.size_SRVF_W])
        q2=tf.reshape(q2, [self.size_batch,  self.size_SRVF_H, self.size_SRVF_W])
        inner_prod= self.Inner(q1, q2)
        dist = tf.acos(tf.clip_by_value(inner_prod,-0.98, 0.98))   ## ds=acos(<q1,q2>)
        return dist
        
    
    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, self.checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ##os.path.join(checkpoint_dir, 'model'),    
        self.saver.save(
            sess=self.session,
            save_path= os.path.join(checkpoint_dir, 'model'),   ###"./save_tst/Model.ckpt" ,
            global_step=self.global_step.eval()
        )

    def load_checkpoint(self, dir):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = dir   ##os.path.join(self.save_dir, self.checkpoint_dir)
        print(checkpoint_dir)
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            #checkpoints_name = 'model-127440'   ###'Model.ckpt-440000'    ##   ##os.path.basename(checkpoints.model_checkpoint_path)
            #checkpoints_name = 'model-81425'  ###'Model.ckpt-440000'    ##   ##os.path.basename(checkpoints.model_checkpoint_path)
            checkpoints_name = 'model-82240'  ###'Model.ckpt-440000'
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False
    

    """def load_checkpoint(self, dir):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = dir ##os.path.join(self.save_dir, self.checkpoint_dir)
        print checkpoint_dir
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False
     """
    

    def py_func(self, func, inp, Tout, stateful=True, name=None, grad=None):    
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))   
        tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
           return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    def y_to_rb_label(self, label):
        number = np.argmax(label)
        one_hot = np.random.uniform(-1, 1, self.rb_dim)
        rb = np.tile(-1*np.abs(one_hot), self.y_dim)
        rb[number * self.rb_dim:(number + 1) * self.rb_dim] = np.abs(one_hot)
        return rb

    def costomLogExp(self, x, name=None):
      with ops.name_scope(name, "ExpLog", [x]) as name:
        LogExp_x = self.py_func(self.log_exp_map,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=self._LogExpGrad)  # <-- here's the call to the gradient
        return LogExp_x[0]    ##tf.reshape(LogExp_x[0], tf.shape(x))

    def _LogExpGrad(self, op, grad):
      return grad
   
    """def custom_test(self, batch_labels, random_seed, dir):
        batch_labels=np.full(64,4)
        if not self.load_checkpoint(dir):
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")
        y_vec = np.zeros(shape=(len(batch_labels), self.y_dim), dtype=np.float32)
        for i, label in enumerate(batch_labels):
            y_vec[i, label-1] = 1

        batch_label_rb = np.zeros(shape=(self.size_batch, self.rb_dim * self.y_dim),
                                           dtype=np.float32)
        for i in range(self.size_batch):
               batch_label_rb[i,:] = self.y_to_rb_label(y_vec[i,:])
        SRVF_generated=tf.reshape(self.exp_fake_data, [self.size_batch,self.size_SRVF_H, self.size_SRVF_W])
        norm=tf.sqrt(self.Inner(SRVF_generated))

        SRVF_generated =self.session.run(SRVF_generated, feed_dict={self.emotion: batch_label_rb})
        print(self.session.run(norm))        
        scipy.io.savemat('generatedSRVF/Happytestes.mat', dict([('x_test', SRVF_generated), ('y_test', batch_labels)]))
      """


    def custom_test(self, batch_labels, random_seed, dir):
        batch_labels=np.full(64,12)  ##START FROM 1-> 12
        if not self.load_checkpoint(dir):
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        y_vec=[]
        for i, label in enumerate(batch_labels):
           l=np.zeros(shape=self.y_dim)
           l[int(label-1)]=1
           y_vec.append(l)

        batch_label_rb = np.zeros(shape=(self.size_batch, self.rb_dim * self.y_dim),
                                           dtype=np.float32)
        for i in range(self.size_batch):
               batch_label_rb[i,:] = self.y_to_rb_label(y_vec[i])

        SRVF_generated=tf.reshape(self.exp_fake, [self.size_batch,self.size_SRVF_H, self.size_SRVF_W])
        norm=tf.sqrt(self.Inner(SRVF_generated, SRVF_generated))

        SRVF_generat =self.session.run(SRVF_generated, feed_dict={self.emotion: batch_label_rb})
        print((self.session.run(norm, feed_dict={self.emotion: batch_label_rb})))
        # print(SRVF_generat[1,:])
        # print ('hhhhhhhhhhhhhhh')
        # print(SRVF_generat[2,:])
        # print ('hhhhhhhhhhhhhhh')
        # print(SRVF_generat[3,:])
        #print_tensors_in_checkpoint_file(file_name='save/Checkpointnewtf_CASIA_CK_concatLabelsChannels/model-277885', tensor_name='', all_tensors=True, all_tensor_names=True)
        scipy.io.savemat('generatedSRVF/SRVF_ForClassification/lr_4/mouth_up.mat', dict([('x_test', SRVF_generat), ('y_test', batch_labels)]))


