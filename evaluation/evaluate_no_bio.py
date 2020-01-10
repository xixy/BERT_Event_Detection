#coding=utf-8
import numpy as np
import sys

def get_chunks(labels):
	"""
	从标签序列中提取chunk
	"""
	chunks = []
	chunk_start, chunk_type = None, None

	for i, label in enumerate(labels):
		if label == 'O':
			# 识别到一个chunk
			if chunk_type != None:
				chunk = (chunk_type, chunk_start, i - 1)
				chunks.append(chunk)
				chunk_start, chunk_type = None, None
			else:
				pass
		# 如果不是O
		else:
			if chunk_type == None:
				chunk_start, chunk_type = i, label
			else:
				# 如果相同的话，那么就是multi-token trigger
				if label == chunk_type:
					continue
				# 否则就是结束，并且是一个新的chunk的开始
				else:
					chunk = (chunk_type, chunk_start, i - 1)
					chunks.append(chunk)
					chunk_start, chunk_type = i, label
	return chunks




def evaluate(result_path):
	"""
	对实验结果进行统计
	"""

	predictions = []
	real_labels = []
	accs = []

	# 1. 读取文件标签
	with open(result_path) as f:
		for line in f:
			contents = line.strip().split('\t')
			if len(contents) == 3:
				real_labels.append(contents[1])
				predictions.append(contents[2])

	# 2. 处理标签

	accs += [a == b for (a, b) in zip(real_labels, predictions)]

	# 2.1 提取chunk
	lab_chunks = set(get_chunks(real_labels))
	lab_pred_chunks = set(get_chunks(predictions))

	# 2.2 进行对比
	# 进行identification的记录
	correct_preds_id, total_correct_id, total_preds_id = 0., 0., 0. 
	# 进行classification的记录
	correct_preds_cl, total_correct_cl, total_preds_cl = 0., 0., 0.

	# 1. 计算identification效果

	# 真实结果
	chunks_without_label = set([(item[1], item[2]) for item in lab_chunks])
	# 预测结果
	chunks_pred_without_label = set([(item[1], item[2]) for item in lab_pred_chunks])
	# 记录正确的chunk数量
	correct_preds_id += len(chunks_without_label & chunks_pred_without_label)
	# 预测出的chunk数量
	total_preds_id += len(chunks_pred_without_label)
	# 正确的chunk数量
	total_correct_id += len(chunks_without_label)

	# 2. 计算classification效果

	# 记录正确的chunk数量
	correct_preds_cl += len(lab_chunks & lab_pred_chunks)
	# 预测出的chunk数量
	total_preds_cl += len(lab_pred_chunks)
	# 正确的chunk数量
	total_correct_cl += len(lab_chunks)

	# 1. 计算trigger identification的结果
	# 计算precision，预测出的chunk中有多少是正确的
	p1 = correct_preds_id / total_preds_id if correct_preds_id > 0 else 0
	# 计算recall，预测正确的chunk占了所有chunk的数量
	r1 = correct_preds_id / total_correct_id if correct_preds_id > 0 else 0
	# 计算f1
	f1_1 = 2 * p1 * r1 / (p1 + r1) if correct_preds_id > 0 else 0

	# 2. 计算trigger classification的结果
	# 计算precision，预测出的chunk中有多少是正确的
	p2 = correct_preds_cl / total_preds_cl if correct_preds_cl > 0 else 0
	# 计算recall，预测正确的chunk占了所有chunk的数量
	r2 = correct_preds_cl / total_correct_cl if correct_preds_cl > 0 else 0
	# 计算f1
	f1_2 = 2 * p2 * r2 / (p2 + r2) if correct_preds_cl > 0 else 0
	# 计算accuracy，用预测对的词的比例来进行表示
	acc = np.mean(accs)

	# 返回结果
	return {
		'p1': 100 * p1,
		'r1': 100 * r1,
		'f1_1': 100 * f1_1,
		'p2': 100 * p2,
		'r2': 100 * r2,
		'f1_2': 100 * f1_2
	}	


if __name__ == '__main__':
	# labels = ['PER', 'PER', 'O', 'O', 'O', 'LOC', 'O']
	# print(get_chunks(labels))
	path = sys.argv[1]
	print(evaluate(path))







