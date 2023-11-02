import tensorflow as tf


from SRVF_WGAN import SRVF_WGAN



flags = tf.compat.v1.flags
flags.DEFINE_integer(flag_name='epoch', default_value=100, docstring='number of epochs')
flags.DEFINE_integer(flag_name='batch_size', default_value=64, docstring='number of batch size')
flags.DEFINE_integer(flag_name='y_dim', default_value=12, docstring='label dimension')

flags.DEFINE_boolean(flag_name='is_train', default_value=False, docstring='training mode')
##flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')

flags.DEFINE_string(flag_name='save_dir', default_value='save', docstring='dir for saving training results')
flags.DEFINE_string(flag_name='test_dir', default_value='test', docstring='dir for testing images')
flags.DEFINE_string(flag_name='checkpoint_dir', default_value='Checkpoint', docstring='dir for loading checkpoints')
FLAGS = flags.FLAGS

def main(_):
    import pprint
    pprint.pprint(FLAGS.__flags)

    config = tf.compat.v1.ConfigProto()
    ###config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as session:
        model = SRVF_WGAN(
            session,
            is_training=FLAGS.is_train,
            save_dir=FLAGS.save_dir,
            ##dataset_name=FLAGS.dataset,
            checkpoint_dir=FLAGS.checkpoint_dir,
            size_batch=FLAGS.batch_size,
            y_dim=FLAGS.y_dim,
        )
        if FLAGS.is_train:
            print ('\n\tTraining Mode')
            model.train(
                num_epochs=8000   ##FLAGS.epoch,  # number of epochs
            )
        else:
            seed = 2019
            print ('\n\tTesting Mode')
            model.custom_test(batch_labels=[], random_seed=seed, dir='save\CheckpointwLoss1_LR_4'
              #batch_labels =[2, 2, 3, 4], random_seed=seed, dir='save\CheckpointwLoss100_LR_6'
              #batch_labels=[], random_seed=seed, dir='save\CheckpointwLoss10_LR_6_geoloss'


            )


if __name__ == '__main__':

    tf.compat.v1.app.run()
