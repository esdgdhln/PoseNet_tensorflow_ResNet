import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('version', 'test_110', '''A version number defining the directory to save
logs and checkpoints''')

tf.app.flags.DEFINE_integer('train_steps', 2000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_integer('batch_size', 50, '''Train batch size''')

tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')
tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want''')


tf.app.flags.DEFINE_string('ckpt_path', 'logs_test_110/model.ckpt-1500', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', True, '''Whether to load a checkpoint and continue
training''')
train_dir = 'logs_' + FLAGS.version + '/'