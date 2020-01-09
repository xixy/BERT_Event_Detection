#coding=utf-8

# 处理层次标签之间的转换等

event_types = {
	'Life':['Be-Born', 'Marry', 'Divorce', 'Injure', 'Die'],
	'Movement':['Transport'],
	'Transaction':['Transfer-Money', 'Transfer-Ownership'],
	'Business':['Start-Org', 'Merge-Org', 'Declare-Bankruptcy', 'End-Org'],
	'Conflict':['Attack', 'Demonstrate'],
	'Contact':['Meet', 'Phone-Write'],
	'Personnel':['Start-Position', 'End-Position', 'Nominate', 'Elect'],
	'Justice':['Arrest-Jail', 'Release-Parole', 'Trial-Hearing', 'Charge-Indict', 'Sue', 'Convict', 'Sentence', 'Fine', 'Execute', 'Extradite', 'Acquit', 'Appeal', 'Pardon']
}


fine2coarse = {'Sue': 'Justice', 'Arrest-Jail': 'Justice', 'Injure': 'Life', \
'Divorce': 'Life', 'Attack': 'Conflict', 'Acquit': 'Justice', 'End-Position': 'Personnel', \
'Demonstrate': 'Conflict', 'Convict': 'Justice', 'Appeal': 'Justice', \
'Trial-Hearing': 'Justice', 'Extradite': 'Justice', 'Declare-Bankruptcy': 'Business', \
'Be-Born': 'Life', 'Phone-Write': 'Contact', 'Start-Org': 'Business', \
'Execute': 'Justice', 'Release-Parole': 'Justice', 'Transfer-Ownership': 'Transaction', \
'Die': 'Life', 'Marry': 'Life', 'Transfer-Money': 'Transaction', 'Meet': 'Contact', \
'Nominate': 'Personnel', 'Start-Position': 'Personnel', 'Elect': 'Personnel', \
'Pardon': 'Justice', 'Sentence': 'Justice', 'Fine': 'Justice', 'End-Org': 'Business', \
'Merge-Org': 'Business', 'Charge-Indict': 'Justice', 'Transport': 'Movement', 'O':'O', '[CLS]':'[CLS]', '[PAD]':'[PAD]'}

fine2root = {'Sue': 'Event', 'Arrest-Jail': 'Event', 'Injure': 'Event', 'Divorce': 'Event', \
'Attack': 'Event', 'Acquit': 'Event', 'End-Position': 'Event', 'Demonstrate': 'Event', \
'Convict': 'Event', 'Appeal': 'Event', 'Trial-Hearing': 'Event', 'Extradite': 'Event', \
'Declare-Bankruptcy': 'Event', 'Be-Born': 'Event', 'Phone-Write': 'Event', 'Start-Org': 'Event', \
'Execute': 'Event', 'Release-Parole': 'Event', 'Transfer-Ownership': 'Event', 'Die': 'Event', \
'Marry': 'Event', 'Transfer-Money': 'Event', 'Meet': 'Event', 'Nominate': 'Event', 'Start-Position': 'Event', \
'Elect': 'Event', 'Pardon': 'Event', 'Sentence': 'Event', 'Fine': 'Event', 'End-Org': 'Event', \
'Merge-Org': 'Event', 'Charge-Indict': 'Event', 'Transport': 'Event', 'O':'O', '[CLS]':'[CLS]', '[PAD]':'[PAD]'}

root_labels = ['Event', 'O', '[CLS]', '[PAD]']
coarse_labels = ['Life',
	'Transaction',
	'Business',
	'Contact',
	'Justice',
	'Personnel',
	'Conflict',
	'Movement',
	'O', 
	'[CLS]', 
	'[PAD]'
	]

fine_labels = ['Acquit', 'Appeal', 'Arrest-Jail', 'Attack', 'Be-Born', 'Charge-Indict', \
	'Convict', 'Declare-Bankruptcy', 'Demonstrate', 'Die', 'Divorce', 'Elect', 'End-Org', \
	'End-Position', 'Execute', 'Extradite', 'Fine', 'Injure', 'Marry', 'Meet', 'Merge-Org', 
	'Nominate', 'O', 'Pardon', 'Phone-Write', 'Release-Parole', 'Sentence', 'Start-Org', \
	'Start-Position', 'Sue', 'Transfer-Money', 'Transfer-Ownership', 'Transport', 'Trial-Hearing', '[CLS]','[PAD]']

root2id = {
	'Event': 0, 
	'O': 1, 
	'[CLS]': 2, 
	'[PAD]': 3
	}


coarse2id={
	'Business': 2,
	'Conflict': 6,
	'Contact': 3,
	'Justice': 4,
	'Life': 0,
	'Movement': 7,
	'O': 8,
	'Personnel': 5,
	'Transaction': 1,
	'[CLS]': 9,
	'[PAD]': 10
	}

fine2id = {
	'Acquit': 0,
	'Appeal': 1,
	'Arrest-Jail': 2,
	'Attack': 3,
	'Be-Born': 4,
	'Charge-Indict': 5,
	'Convict': 6,
	'Declare-Bankruptcy': 7,
	'Demonstrate': 8,
	'Die': 9,
	'Divorce': 10,
	'Elect': 11,
	'End-Org': 12,
	'End-Position': 13,
	'Execute': 14,
	'Extradite': 15,
	'Fine': 16,
	'Injure': 17,
	'Marry': 18,
	'Meet': 19,
	'Merge-Org': 20,
	'Nominate': 21,
	'O': 22,
	'Pardon': 23,
	'Phone-Write': 24,
	'Release-Parole': 25,
	'Sentence': 26,
	'Start-Org': 27,
	'Start-Position': 28,
	'Sue': 29,
	'Transfer-Money': 30,
	'Transfer-Ownership': 31,
	'Transport': 32,
	'Trial-Hearing': 33,
	'[CLS]': 34,
	'[PAD]': 35
	}

# 36 * 4
fine2rootmapper = [
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[0, 1, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[1, 0, 0, 0], 
[0, 0, 1, 0], 
[0, 0, 0, 1]
]

fine2coarsemapper = [
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]

