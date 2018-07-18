import tensorflow as tf
import tensorlayer as tl
import argparse
from data.mx2tfrecords import parse_function
import os
# from nets.L_Resnet_E_IR import get_resnet
# from nets.L_Resnet_E_IR_GBN import get_resnet
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
from losses.face_losses import arcface_loss
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test
from mobilenets import mobilenet_v2_new
import numpy as np

slim = tf.contrib.slim


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=100, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000, help='epoch to train the network')
    parser.add_argument('--batch_size', default=256, help='batch size to train network')
    parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='/root/upload', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=81161, help='the image size')
    parser.add_argument('--tfrecords_file_path', default='/root/upload', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/MobileDistRes-noisy-07170921/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/MobileDistRes-noisy-07170921/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/MobileDistRes-noisy-07170921/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=100, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=500, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    parser.add_argument('--dim_eb', default=512, help='dimensions of the output embedding')
    parser.add_argument('--tau', dest='tau', default=3.0, help='KD method. tau stands for temperature.', type=float)
    parser.add_argument('--lamda', dest='lamda', default=0.3, help='KD method. lamda between original loss and soft-target loss.', type=float)
    parser.add_argument('--teacher_model_path', default='./pretrained/InsightFace_iter_175000.ckpt', type=str, help='restored teacher model path')
    parser.add_argument('--student_model_path', default='./output/MobileDistRes-noisy-07151950/ckpt/MobileDistRes-noisy_iter_35600.ckpt', type=str, help='restored student model path')
    parser.add_argument('--gpus', default=1)

    parser.add_argument('--noisy_ratio', dest='Nratio', default=0.5, help="noisy ratio", type=float)
    parser.add_argument('--noisy_sigma', dest='Nsigma', default=0.9, help="noisy sigma", type=float)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    tf.reset_default_graph()
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    
    # trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    
    tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran_star_merge_with_ms1m.tfrecords')
    
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)
    

    graph_t = tf.Graph()
    with graph_t.as_default():
        dataset = tf.data.TFRecordDataset(tfrecords_f)
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(buffer_size=args.buffer_size)
        dataset = dataset.batch(args.batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        dropout_keep_rate = tf.placeholder(name='dropout_keep_rate', dtype=tf.float32)
        images_t = tf.placeholder(name='img_inputs_t', shape=[None, *args.image_size, 3], dtype=tf.float32)
        labels_t = tf.placeholder(name='img_labels_t', shape=[None, ], dtype=tf.int64)

        t_model_path = args.teacher_model_path
        w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
        net_t = get_resnet(images_t, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_keep_rate) 
        embedding_tensor_t = net_t.outputs
    
        logit_t_ori = arcface_loss(embedding=embedding_tensor_t, labels=labels_t, var_scope='arcface_loss', w_init=w_init_method, out_num=args.num_output)
        # t_tau = tf.scalar_mul(1.0/args.tau, logit_t)
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # print(tf.shape(t_tau))
        # print(t_tau)
        # 
        drop_scale = 1/args.Nratio
        noisy_mask = tf.nn.dropout( tf.constant(np.float32(np.ones((args.batch_size,1)))/drop_scale) ,keep_prob=args.Nratio) #(batchsize,1)
        gaussian = tf.random_normal(shape=[args.batch_size,1], mean=0.0, stddev=args.Nsigma)
        noisy = tf.multiply(noisy_mask, gaussian)
        noisy_add = tf.add(tf.constant(np.float32(np.ones((args.batch_size,1)))), noisy)
        # logit_t = tf.mul(logit_t_ori, tf.tile(noisy, tf.constant([1,args.num_output])))   #(batchsize,10)
        logit_t = tf.add(logit_t_ori, tf.tile(noisy, tf.constant([1,args.num_output])))
        
        pred_t = tf.nn.softmax(logit_t_ori)
        acc_t = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_t, axis=1), labels_t), dtype=tf.float32))

        print('########################')
        print('teacher model start to restore')
        saver_restore_t = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
        config.gpu_options.allow_growth = True
        sess_t = tf.Session(config=config)
        sess_t.run(tf.global_variables_initializer())
        saver_restore_t.restore(sess_t, t_model_path)
        print('########################')
        print('teacher model resnet 101 restored')


    graph_s = tf.Graph()
    with graph_s.as_default():
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
        
        logit_t_input = tf.placeholder(name='teacher_input_to_student', shape=[None, args.num_output], dtype=tf.float32)

        images_s = tf.placeholder(name='img_inputs_s', shape=[None, *args.image_size, 3], dtype=tf.float32)
        labels_s = tf.placeholder(name='img_labels_s', shape=[None, ], dtype=tf.int64)

        s_model_path = args.student_model_path

        with slim.arg_scope(mobilenet_v2_new.mobilenet_v2_arg_scope()):
            net_s, endpoints = mobilenet_v2_new.mobilenet_v2(images_s, num_classes=None, is_training=True, dropout_keep_prob=0.5)
        with tf.variable_scope('Logits'):
            # net_s = slim.conv2d(net_s, 512, [4, 4], padding='VALID', scope='conv_last')
            net_s = tf.squeeze(net_s, axis=[1,2])
            # net_student = slim.dropout(net_student, keep_prob=0.4, scope='dropout_last')
            net_s = slim.fully_connected(net_s, num_outputs=args.dim_eb, activation_fn=None, scope='fc_last')
        with slim.arg_scope(mobilenet_v2_new.mobilenet_v2_arg_scope()):
            net_test_s, _ = mobilenet_v2_new.mobilenet_v2(images_s, num_classes=None, is_training=False, dropout_keep_prob=1.0, reuse = True, scope='MobilenetV2')
        net_test_s = tf.squeeze(net_test_s, axis=[1,2])
        embedding_tensor_s = slim.fully_connected(net_test_s, num_outputs=args.dim_eb, activation_fn=None, reuse=True, scope='Logits/fc_last')
        
        # checkpoint_exclude_scopes = 'Logits'
        # exclusions = None
        # if checkpoint_exclude_scopes:
        #     exclusions = [
        #         scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        # variables_to_restore = []
        # for var in slim.get_model_variables():
        #     excluded = False
        #     for exclusion in exclusions:
        #         if var.op.name.startswith(exclusion):
        #             excluded = True
        #     if not excluded:
        #         variables_to_restore.append(var)
        print('################################')
        print('variables in student model to restore are set')

        logit_s = arcface_loss(embedding=net_s, labels=labels_s, var_scope='arcface_loss_s', w_init=w_init_method, out_num=args.num_output)
        # s_tau = tf.scalar_mul(1.0/args.tau, logit_s)
        # soft_loss_logit = tf.nn.l2_loss(logit_s-logit_t_input)/args.batch_size
        soft_loss_logit = tf.reduce_mean(tf.scalar_mul(0.5, tf.square(logit_s-logit_t_input)))
        # inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_s, labels=labels_s))
        total_loss = soft_loss_logit

        p = int(512.0/args.batch_size)
        lr_steps = [(p*val-35600) for val in args.lr_steps]
        print(lr_steps)
        lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')

    # tl.layers.set_name_reuse(True)
    # test_net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, reuse=True, keep_rate=dropout_rate)
    # embedding_tensor = test_net.outputs

        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)
        grads = opt.compute_gradients(total_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads, global_step=global_step)
        
        pred_s = tf.nn.softmax(logit_s)
        acc_s = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_s, axis=1), labels_s), dtype=tf.float32))

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
        config.gpu_options.allow_growth = True
        sess_s = tf.Session(config=config)
    

        summary = tf.summary.FileWriter(args.summary_path, sess_s.graph)
        summaries = []

        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        
        summaries.append(tf.summary.scalar('soft_loss_logit', soft_loss_logit))
        summaries.append(tf.summary.scalar('total_loss', total_loss))

        summaries.append(tf.summary.scalar('leraning_rate', lr))
        summary_op = tf.summary.merge(summaries)
    
        # saver_restore_s = tf.train.Saver(var_list=variables_to_restore)
        saver_restore_s = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.saver_maxkeep)

        sess_s.run(tf.global_variables_initializer())
    
        print('########################')
        print('start restoring student model')
        saver_restore_s.restore(sess_s, s_model_path)
        print('########################')
        print('student model Mobilenet restored')
    


#########################################################################################
##               training and validating
#########################################################################################


    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')

    count = 0
    total_accuracy = {}

    for i in range(args.epoch):
        sess_t.run(iterator.initializer)
        while True:
            try:
                images_train, labels_train = sess_t.run(next_element)
                feed_dict_t = {images_t: images_train, labels_t: labels_train, dropout_keep_rate: 1.0}
                feed_dict_t.update(tl.utils.dict_to_one(net_t.all_drop))
                start = time.time()
                logit_t_out, acc_t_val = sess_t.run([logit_t, acc_t], feed_dict=feed_dict_t)
                
                feed_dict_s = {images_s: images_train, labels_s: labels_train, logit_t_input: logit_t_out}
                _, total_loss_val, soft_loss_val, _, acc_s_val = \
                    sess_s.run([train_op, total_loss, soft_loss_logit, inc_op, acc_s],
                              feed_dict=feed_dict_s)
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss is %.2f , soft '
                          'loss is %.2f, training accuracy is %.6f, teacher accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_val, soft_loss_val, acc_s_val, acc_t_val, pre_sec))
                count += 1

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    # feed_dict = {images: images_train, labels: labels_train, teacher_tau_input: tau_input}
                    # feed_dict.update(net_student.all_drop)
                    summary_op_val = sess_s.run(summary_op, feed_dict=feed_dict_s)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'MobileDistRes-noisy_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess_s, filename)

                # validate
                if count > 0 and count % args.validate_interval == 0:
                    # feed_dict_test_t = {dropout_keep_rate: 1.0}
                    # feed_dict_test_t.update(tl.utils.dict_to_one(net_t.all_drop))
                    # results_t = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess_t,
                    #          embedding_tensor=embedding_tensor_t, batch_size=args.batch_size, feed_dict=feed_dict_test_t,
                    #          input_placeholder=images_t)
                    # print('test accuracy of teacher model is: ', str(results_t[0]))

                    feed_dict_test_s = {}
                    results_s = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess_s,
                             embedding_tensor=embedding_tensor_s, batch_size=args.batch_size, feed_dict=feed_dict_test_s,
                             input_placeholder=images_s)
                    print('test accuracy of student model is: ', str(results_s[0]))

                    total_accuracy[str(count)] = results_s[0]
                    log_file.write('########'*10+'\n')
                    log_file.write(','.join(list(total_accuracy.keys())) + '\n')
                    log_file.write(','.join([str(val) for val in list(total_accuracy.values())])+'\n')
                    log_file.flush()
                    if max(results_s) > 0.996:
                        print('best accuracy is %.5f' % max(results_s))
                        filename = 'MobileDistRes-noisy_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess_s, filename)
                        log_file.write('######Best Accuracy######'+'\n')
                        log_file.write(str(max(results_s))+'\n')
                        log_file.write(filename+'\n')

                        log_file.flush()
            # except tf.errors.OutOfRangeError:
            except:
                print("End of epoch %d" % i)
                break
    log_file.close()
    log_file.write('\n')