from data.vocab import TextEncoder
from data.dataset import create_attention_mask
from transformer.model import load_openai_model, K
from openai.train import dropout, tf, embed, block, find_trainable_variables, json, np

n_vocab = 40478
n_ctx = 7
n_embd = 768
embd_pdrop = 0.1
n_layer = 12
n_batch_train = 2
n_transfer = 1 + 12 * 12
n_head = 12


def model(X, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [n_vocab + TextEncoder.SPECIAL_COUNT + n_ctx, n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)
        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d' % layer, train=train, scale=True)
        return h


X_train = tf.placeholder(tf.int32, [n_batch_train, n_ctx, 2])
X = tf.placeholder(tf.int32, [None, 2, n_ctx, 2])
res = model(X_train)

params = find_trainable_variables('model')
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())

with open('openai/model/params_shapes.json') as f:
    shapes = json.load(f)
offsets = np.cumsum([np.prod(shape) for shape in shapes])
init_params = [np.load('openai/model/params_{}.npy'.format(n)) for n in range(10)]
init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
init_params[0] = init_params[0][:n_ctx]
init_params[0] = np.concatenate(
    [init_params[1], (np.random.randn(TextEncoder.SPECIAL_COUNT, n_embd) * 0.02).astype(np.float32), init_params[0]], 0)
del init_params[1]

sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])
xmb = np.random.randint(0, n_vocab, (n_batch_train, n_ctx, 2))
xmb[:, :, 1] = np.random.randint(0, n_ctx, (n_batch_train, n_ctx))
xmb_tf = xmb.copy()
xmb_tf[:, :, 1] += n_vocab + TextEncoder.SPECIAL_COUNT
tf_result = sess.run([res], {X_train: xmb_tf})

keras_model = load_openai_model(ignore_mask=False, use_one_embedding_dropout=False, use_decoder_bias=False, debug=False,
                                max_len=7)
K.set_learning_phase(0)
mask = create_attention_mask(None, True, n_batch_train, n_ctx)
k_result = keras_model.predict(
    [xmb[:, :, 0], np.zeros((n_batch_train, n_ctx), dtype=np.int64), xmb[:, :, 1], mask])

print((tf_result[0] == k_result).all())
