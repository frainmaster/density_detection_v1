import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import xlsxwriter as xl
from PIL import Image
import math

def get_pairs(a):
	b = []
	for i in range(len(a)-1):
		c = a.pop(0)
		for j in a:
			b.append([c, j])
	return b

def getDistance(a, b):
	dist = math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
	return dist

def density_detection(input_video, detection_graph, category_index, start_point, end_point):

	total_passed_object = 0
	# output path
	output_path = 'output/'
	# input video
	cap = cv2.VideoCapture(input_video)

	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	vid_name = input_video[input_video.rfind('/')+1: input_video.rfind('.')]

	print('height: ' + str(height))
	print('width: ' + str(width))
	print('frame count: ' + str(total_frame))
	print('fps: ' + str(fps))
	print('video name: ' + vid_name)

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	output_video = cv2.VideoWriter(output_path + vid_name + '.mp4', fourcc, fps, (width, height))

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			frame_counter = 1
			# variables to work with excel
			total_passed_object_per_interval = 0
			total_passed_object = 0
			sop_violation = 0
			current_sop_violation = 0
			total_sop_violation = 0
			sop_violation_timer = 0
			interval_duration = 5 # the interval to write in excel file, in seconds
			final_col = 0
			current_row = 0

			workbook = xl.Workbook(output_path + vid_name + '_counter.xlsx')
			worksheet = workbook.add_worksheet()
			bold = workbook.add_format({'bold': True})

			# write table header
			worksheet.write('A1', 'TIME (S)', bold)
			worksheet.write('B1', 'COUNT', bold)
			worksheet.write('C1', 'CUMUL. COUNT', bold)
			worksheet.write('D1', 'SOP VIOLATION', bold)
			worksheet.write('E1', 'CUMUL. SOP VIOLATION', bold)

			# for all the frames that are extracted from input video
			while(cap.isOpened()):
				ret, frame = cap.read()

				if not ret:
					print('end of the video file...')
					break

				# draw rectangle; yellow, line thickness=1
				(w1, h1), (w2, h2) = start_point, end_point
				cv2.rectangle(frame, start_point, end_point, (0, 255, 255), 2)
				cropped_frame = frame[h1:h2, w1:w2]
				rec_width = w2 - w1
				rec_height = h2 - h1
				if w1 == h1 == w2 == h2:
					cropped_frame = frame
					rec_width = width
					rec_height = height

				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(cropped_frame, axis=0)

				# Actual detection.
				(boxes, scores, classes, num) = sess.run(
					[detection_boxes, detection_scores, detection_classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})

				# insert information text to video frame
				font = cv2.FONT_HERSHEY_SIMPLEX

				counter, csv_line, result = vis_util.visualize_boxes_and_labels_on_image_array(
					cap.get(1),
					cropped_frame,
					1,
					0,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					targeted_objects='person',
					use_normalized_coordinates=True,
					min_score_thresh=.55,
					line_thickness=4)

				# draw line between 2 or more people
				if total_passed_object_per_interval > 1:
					mid_points = []
					for a, b in enumerate(boxes[0]):
						if scores[0][a] > .6:
							mid_x = (boxes[0][a][3] + boxes[0][a][1])/2
							mid_y = (boxes[0][a][2] + boxes[0][a][0])/2
							mid_points.append((int(mid_x*rec_width) + w1, int(mid_y*rec_height) + h1))
					
					for i in get_pairs(mid_points):
						# check if line is less than 1 meter
						print('LINE DRAWN at points {}, {}'.format(str(i[0]), str(i[1])))
						meter = 100
						if getDistance(i[0], i[1]) < meter:
							sop_violation = len(mid_points)
							if sop_violation > 0:
								current_sop_violation = sop_violation
							if frame_counter-sop_violation_timer > fps:
								total_sop_violation += sop_violation
								if total_sop_violation == 1:
									total_sop_violation = 2
							cv2.line(frame, i[0], i[1], (0, 0, 255) , 2)
							sop_violation_timer = frame_counter
						else:
							cv2.line(frame, i[0], i[1], (0, 255, 0) , 2)

				cv2.putText(
					frame,
					'People count: ' + str(total_passed_object),
					(10, 35),
					font,
					0.6,
					(0,255,255),
					2,
					cv2.FONT_HERSHEY_SIMPLEX)

				cv2.putText(
					frame,
					'People violate: ' + str(total_sop_violation),
					(10, 65),
					font,
					0.6,
					(0,255,255),
					2,
					cv2.FONT_HERSHEY_SIMPLEX)

				if len(result) != 0:
					cv2.putText(
						frame,
						'Current: ' + result[result.rfind(':')+2:],
						(10, 95),
						font,
						0.6,
						(0,255,255),
						2,
						cv2.FONT_HERSHEY_SIMPLEX)
					# if frame_counter % fps == 0:
						# total_passed_object_per_interval += int(result[result.rfind(':')+2:])
						# if int(result[result.rfind(':')+2:]) > total_passed_object_per_interval:
					total_passed_object_per_interval = int(result[result.rfind(':')+2:])

				output_video.write(frame)
				print ('writing frame ' + str(frame_counter) + '/' + str(total_frame))
				print(result[result.rfind(':')+2:])

				# writing to excel file
				if frame_counter % (interval_duration*fps) == 0:
					total_passed_object += total_passed_object_per_interval
					current_row = frame_counter//(interval_duration*fps)
					worksheet.write(current_row, 0, frame_counter//fps)
					worksheet.write(current_row, 1, total_passed_object_per_interval)
					worksheet.write(current_row, 3, current_sop_violation)
					if current_row == 1:
						worksheet.write(current_row, 2, '=B2')
						worksheet.write(current_row, 4, '=D2')
					else:
						worksheet.write(current_row, 2, '=B' + str(current_row+1) + '+C' + str(current_row)) # =B(x+1)+C(x)
						worksheet.write(current_row, 4, '=D' + str(current_row+1) + '+E' + str(current_row)) # =B(x+1)+C(x)
					total_passed_object_per_interval = 0
					current_sop_violation = 0

				if frame_counter == total_frame:
					final_col = current_row

				frame_counter += 1

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			# crate the graph consisting of bar and line chart
			bar_chart = workbook.add_chart({'type':'column'})
			bar_chart.add_series({
				'name':'=Sheet1!B1',
				'categories':'=Sheet1!A2:A' + str(final_col+1),
				'values':'=Sheet1!B2:B' + str(final_col+1)
				})
			bar_chart.add_series({
				'name':'=Sheet1!D1',
				'categories':'=Sheet1!A2:A' + str(final_col+1),
				'values':'=Sheet1!D2:D' + str(final_col+1)
				})

			# line_chart = workbook.add_chart({'type':'line'})
			# line_chart.add_series({
			# 	'name':'=Sheet1!C1',
			# 	'categories':'=Sheet1!A2:A' + str(final_col+1),
			# 	'values':'=Sheet1!C2:C' + str(final_col+1)
			# 	})
			# bar_chart.combine(line_chart)
			line_chart_2 = workbook.add_chart({'type':'line'})
			line_chart_2.add_series({
				'name':'=Sheet1!E1',
				'categories':'=Sheet1!A2:A' + str(final_col+1),
				'values':'=Sheet1!E2:E' + str(final_col+1)
				})
			bar_chart.combine(line_chart_2)

			bar_chart.set_title({'name':'Pedestrians Counting and SOP Violations'})
			bar_chart.set_x_axis({'name':'=Sheet1!A1'})
			bar_chart.set_y_axis({'name':'No of Pedestrians'})
			worksheet.insert_chart('H2', bar_chart)

			# mode, median and mean of data (count)
			worksheet.write('A' + str(final_col+3), 'MODE', bold)
			worksheet.write('A' + str(final_col+4), 'MEDIAN', bold)
			worksheet.write('A' + str(final_col+5), 'MEAN', bold)
			worksheet.write('A' + str(final_col+6), 'SD', bold)

			worksheet.write('B' + str(final_col+3), '=MODE(B2:B' + str(final_col+1) + ')')
			worksheet.write('B' + str(final_col+4), '=MEDIAN(B2:B' + str(final_col+1) + ')')
			worksheet.write('B' + str(final_col+5), '=AVERAGE(B2:B' + str(final_col+1) + ')')
			worksheet.write('B' + str(final_col+6), '=STDEV(B2:B' + str(final_col+1) + ')')

			worksheet.write('C' + str(final_col+3), '=MODE(C2:C' + str(final_col+1) + ')')
			worksheet.write('C' + str(final_col+4), '=MEDIAN(C2:C' + str(final_col+1) + ')')
			worksheet.write('C' + str(final_col+5), '=AVERAGE(C2:C' + str(final_col+1) + ')')
			worksheet.write('C' + str(final_col+6), '=STDEV(C2:C' + str(final_col+1) + ')')

			worksheet.write('D' + str(final_col+3), '=MODE(D2:D' + str(final_col+1) + ')')
			worksheet.write('D' + str(final_col+4), '=MEDIAN(D2:D' + str(final_col+1) + ')')
			worksheet.write('D' + str(final_col+5), '=AVERAGE(D2:D' + str(final_col+1) + ')')
			worksheet.write('D' + str(final_col+6), '=STDEV(D2:D' + str(final_col+1) + ')')

			worksheet.write('E' + str(final_col+3), '=MODE(E2:E' + str(final_col+1) + ')')
			worksheet.write('E' + str(final_col+4), '=MEDIAN(E2:E' + str(final_col+1) + ')')
			worksheet.write('E' + str(final_col+5), '=AVERAGE(E2:E' + str(final_col+1) + ')')
			worksheet.write('E' + str(final_col+6), '=STDEV(E2:E' + str(final_col+1) + ')')

			workbook.close()

			cap.release()
			cv2.destroyAllWindows()