import ROOT 
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
#cbmsim->Draw("MCTrack.fStartZ","(MCTrack.fStartZ <= -7070)")
parser = argparse.ArgumentParser()

parser.add_argument('-n', action='store', dest='file_number', type=int)

results = parser.parse_args()

file_number = results.file_number

f = ROOT.TFile.Open("/eos/experiment/ship/data/Mbias/background-prod-2018/pythia8_Geant4_10.0_withCharmandBeauty%d000_mu.root"%file_number)

event_number_save = 0

total_saved = np.zeros(4)

number_muons_wanted = 1000000

t0 = time.time()

time_info = np.empty((0,2))

for fragment in range(0, 20):
	print(fragment)

	# print('FRAGMENT',fragment, 'with event_number_save',event_number_save)

	sample_0_0_pos13 = np.empty((0,5))
	sample_x_x_pos13 = np.empty((0,7))
	sample_0_0_neg13 = np.empty((0,5))
	sample_x_x_neg13 = np.empty((0,7))

	list_veto_pos13 = np.empty((0,2))
	list_veto_neg13 = np.empty((0,2))

	event_number = -1
	length = np.zeros(2)
	size = 10000

	for events in f.cbmsim:
		event_number += 1

		if length[0] >= size and length[1] >= size:
			break

		if event_number > event_number_save:

			# if event_number % 9999 == 0: print(event_number/4.978E7, length)

			for e in events.vetoPoint:
				if e.PdgCode() == 13:
					list_veto_pos13 = np.append(list_veto_pos13, [[event_number, e.GetTrackID()]], axis=0)
					length[0] += 1
				if e.PdgCode() == -13:
					list_veto_neg13 = np.append(list_veto_neg13, [[event_number, e.GetTrackID()]], axis=0)
					length[1] += 1

	event_number_save = event_number

	event_number = -1
	event_index_in_list = -1
	quit_out_loop = False

	for events in f.cbmsim:

		if quit_out_loop == True: break

		event_number += 1
		track_number = -1

		if event_number in list_veto_pos13[:,0]:

			event_index_in_list += 1

			for e in events.MCTrack:

				track_number += 1

				if track_number == list_veto_pos13[event_index_in_list, 1]:

					#fill here
					if e.GetStartX() == 0 and e.GetStartY() == 0:
						sample_0_0_pos13 = np.append(sample_0_0_pos13, [[e.GetWeight()/768.75,e.GetStartZ(), e.GetPx(), e.GetPy(), e.GetPz()]], axis=0)
					else:
						sample_x_x_pos13 = np.append(sample_x_x_pos13, [[e.GetWeight()/768.75,e.GetStartX(),e.GetStartY(),e.GetStartZ(), e.GetPx(), e.GetPy(), e.GetPz()]], axis=0)

					buffer_event_number = list_veto_pos13[event_index_in_list, 0]

					if event_index_in_list+1 < len(list_veto_pos13):
						if list_veto_pos13[event_index_in_list+1,0] == buffer_event_number:
							event_index_in_list += 1

					if event_index_in_list == len(list_veto_pos13)-1:
						quit_out_loop = True


	# print(np.shape(sample_0_0_pos13), np.shape(sample_x_x_pos13))

	np.save('sample_save/pos13_0_0/fragment_%d_%d'%(file_number,fragment), sample_0_0_pos13)
	np.save('sample_save/pos13_x_x/fragment_%d_%d'%(file_number,fragment), sample_x_x_pos13)


	event_number = -1
	event_index_in_list = -1
	quit_out_loop = False

	for events in f.cbmsim:

		if quit_out_loop == True: break

		event_number += 1
		track_number = -1

		if event_number in list_veto_neg13[:,0]:

			event_index_in_list += 1

			for e in events.MCTrack:

				track_number += 1

				if track_number == list_veto_neg13[event_index_in_list, 1]:

					if e.GetStartX() == 0 and e.GetStartY() == 0:
						sample_0_0_neg13 = np.append(sample_0_0_neg13, [[e.GetWeight()/768.75,e.GetStartZ(), e.GetPx(), e.GetPy(), e.GetPz()]], axis=0)
					else:
						sample_x_x_neg13 = np.append(sample_x_x_neg13, [[e.GetWeight()/768.75,e.GetStartX(),e.GetStartY(),e.GetStartZ(), e.GetPx(), e.GetPy(), e.GetPz()]], axis=0)

					buffer_event_number = list_veto_neg13[event_index_in_list, 0]

					if event_index_in_list+1 < len(list_veto_neg13):
						if list_veto_neg13[event_index_in_list+1,0] == buffer_event_number:
							event_index_in_list += 1

					if event_index_in_list == len(list_veto_neg13)-1:
						quit_out_loop = True


	# print(np.shape(sample_0_0_neg13), np.shape(sample_x_x_neg13))

	np.save('sample_save/neg13_0_0/fragment_%d_%d'%(file_number,fragment), sample_0_0_neg13)
	np.save('sample_save/neg13_x_x/fragment_%d_%d'%(file_number,fragment), sample_x_x_neg13)

	min_max_0_0_pos13 = np.empty((4,2))
	min_max_x_x_pos13 = np.empty((6,2))
	min_max_0_0_neg13 = np.empty((4,2))
	min_max_x_x_neg13 = np.empty((6,2))

	for x in range(0, 4):
		min_max_0_0_pos13[x][0] = np.amin(sample_0_0_pos13[:,x+1])
		min_max_0_0_pos13[x][1] = np.amax(sample_0_0_pos13[:,x+1])

		min_max_0_0_neg13[x][0] = np.amin(sample_0_0_neg13[:,x+1])
		min_max_0_0_neg13[x][1] = np.amax(sample_0_0_neg13[:,x+1])

	for x in range(0, 6):
		min_max_x_x_pos13[x][0] = np.amin(sample_x_x_pos13[:,x+1])
		min_max_x_x_pos13[x][1] = np.amax(sample_x_x_pos13[:,x+1])

		min_max_x_x_neg13[x][0] = np.amin(sample_x_x_neg13[:,x+1])
		min_max_x_x_neg13[x][1] = np.amax(sample_x_x_neg13[:,x+1])


	np.save('sample_save/pos13_0_0/min_max_%d_%d'%(file_number,fragment), min_max_0_0_pos13)
	np.save('sample_save/pos13_x_x/min_max_%d_%d'%(file_number,fragment), min_max_x_x_pos13)

	np.save('sample_save/neg13_0_0/min_max_%d_%d'%(file_number,fragment), min_max_0_0_neg13)
	np.save('sample_save/neg13_x_x/min_max_%d_%d'%(file_number,fragment), min_max_x_x_neg13)

	total_saved += [np.shape(sample_0_0_pos13)[0],np.shape(sample_x_x_pos13)[0],np.shape(sample_0_0_neg13)[0],np.shape(sample_x_x_neg13)[0]]

	# print(total_saved, np.sum(total_saved))

	t1 = time.time()

	total = t1-t0

	time_info = np.append(time_info, [[fragment,total]],axis=0)

	plt.plot(time_info[:,0],time_info[:,1])
	plt.savefig('sample_save/time.png')
	plt.close('all')
	
quit()















