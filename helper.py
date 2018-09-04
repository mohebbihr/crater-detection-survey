import numpy as np
import math
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def calculateDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def delete_duplicate_keep_best(dataframe):
    # Note this method now takes a dataframe as input
    
    if len(dataframe) < 2:
        # nothing to do
        return dataframe

    Crater_data = dataframe
    # extract axes
    x = Crater_data[0].values.tolist()
    y = Crater_data[1].values.tolist()
    s = Crater_data[2].values.tolist()
    p = Crater_data[3].values.tolist()
    Points = []
    while len(x) > 0:
        # a group is a set of similar points
        Group_x = [x[0]]
        Group_y = [y[0]]
        Group_s = [s[0]]
        Group_p = [p[0]]
        index = [0]
        for i in range(1,len(x)):
            d_current = calculateDistance(x[0],y[0],x[i],y[i])

            # accept in group only if 
            d = min(s[0], s[i])
            if d_current <= d and s[i] < 2*d:
                Group_x.append(x[i])
                Group_y.append(y[i])
                Group_s.append(s[i])
                Group_p.append(p[i])
                index.append(i)
        # after group is defined, extract its elements from list
        x = list(np.delete(x,index))
        y = list(np.delete(y,index))
        s = list(np.delete(s,index))
        p = list(np.delete(p,index))
        Points.append([Group_x,Group_y,Group_s, Group_p])

    # now reduce groups
    center_size = []
    for i, (Xs, Ys, Ss, Ps) in enumerate(Points):
        # we take the point with best prediction confidence
        best_index = np.argmax(Ps)
        x_center = Xs[best_index]
        y_center = Ys[best_index]
        size = Ss[best_index]
        prob = Ps[best_index]
        center_size += [[x_center,y_center,size, prob]]

    return pd.DataFrame(center_size)

def save_gt(img,gt, save_path):
    nseg = 64    
    implot = plt.imshow(img)
    for i in range(0, len(gt)):
        x = gt[i,0]
        y = gt[i,1]
        r = gt[i,2]
        
        theta = np.linspace(0.0, (2 * math.pi), (nseg + 1))
        pline_x = np.add(x, np.dot(r, np.cos(theta)))
        pline_y = np.add(y, np.dot(r, np.sin(theta)))
        plt.plot(pline_x, pline_y, 'b-')
    
    plt.savefig(save_path +'_gt.png', bbox_inches='tight', dpi=400)
    plt.show()

def draw_craters_circles(img_path, dataframe, show_probs=True):
    img_filename = os.path.basename(img_path)
    img = cv.imread(img_path, 1)
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)

    color = (0, 255, 0)
    font = cv.FONT_HERSHEY_SIMPLEX
    
    for index, row in dataframe.iterrows():
        winS = int(row[2])
        r = int(winS/2)
        x = int(row[0])
        y = int(row[1])
        # if we want to see where is processed.
        cv.circle(img, (x, y), r, color, 2)
        if show_probs:
            cv.putText(img, "%f" % row[3], (x, y-5), font, 0.6, color, 2)

    return img

    
def draw_craters_rectangles(img_path, dataframe, show_probs=True):
    img_filename = os.path.basename(img_path)
    img = cv.imread(img_path, 1)
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)

    color = (0, 255, 0)
    font = cv.FONT_HERSHEY_SIMPLEX
    
    for index, row in dataframe.iterrows():
        winS = int(row[2])
        half_winS = int(winS/2)
        x = int(row[0] - half_winS)
        y = int(row[1] - half_winS)
        # if we want to see where is processed.
        cv.rectangle(img, (x, y), (x + winS, y + winS), color, 2)
        if show_probs:
            cv.putText(img, "%f" % row[3], (x, y-5), font, 0.6, color, 2)

    return img

def isamatch(x_gt, y_gt, r_gt, x_dt, y_dt, r_dt, param):
    
    match_ratio = abs(x_gt-x_dt) + abs(y_gt-y_dt) + 2*abs(r_gt-r_dt)
    
    if (((x_gt-x_dt)^2 + (y_gt-y_dt)**2 > (param.xy_tol + r_gt*0.2)**2) or (abs(r_gt-r_dt) > param.d_tol*r_gt) or match_ratio > 100 ):
        return False    
    
    return True

def ismember(a_vec, b_vec):        
    booleans = np.in1d(a_vec, b_vec)
    return booleans

def plot_dacc(r_tp,r_fp,r_fn, save_path, param):
    
    d_list = []
    dr_list = []
    fr_list = []
    bf_list = []
    qr_list = []
    fm_list = []
    
    for d in range(param.dmin, 41):
        tp = sum(np.dot(2, r_tp) >= d)
        fp = sum(np.dot(2, r_fp) >= d)
        fn = sum(np.dot(2, r_fn) >= d)
        
        dr = tp/(tp + fn)
        fr = fp / (tp + fp)
        bf = fp / tp
        qr = tp / (tp + fp + fn)
        f_measure = 2*tp/(2*tp+fp+fn)
        
        d_list.append(d)
        dr_list.append(dr)
        fr_list.append(fr)
        bf_list.append(bf)
        qr_list.append(qr)
        fm_list.append(f_measure)
    
    dr_list = np.dot(100, dr_list)    
    fr_list = np.dot(100, fr_list)    
    bf_list = np.dot(100, bf_list)    
    qr_list = np.dot(100, qr_list)
    fm_list = np.dot(100, fm_list) 
       
    plt.plot(d_list, dr_list, 'go', d_list, fr_list, 'y.', d_list, bf_list, 'r*', d_list, qr_list, 'bx', d_list, fm_list, 'c^')
    plt.savefig(save_path +'_dacc.png', bbox_inches='tight', dpi=400)
    plt.show()
    
# testset_path: the name of testfile
# craters: a list of detected craters contains x,y,r
# gt: ground truth loaded from csv files. it must be a dataframe
# img: image of testset_name

def evaluate(craters, gt, img, nseg, save_figs, save_path, param):
        
    #sort by radius
    gt = gt.sort_values(by=[2]).values
    dt = craters.sort_values(by=[2]).values
    
    gt_visit = np.zeros(len(gt), dtype=int)
    dt_visit = np.zeros(len(dt), dtype=int)
    
    # number of correct positive predictions
    p = 0
    errors = []
    match_index = []
    
    for v in range(0,len(gt)):
        x_gt = gt[v][0]
        y_gt = gt[v][1]
        r_gt = gt[v][2]
        
        if (not ((2*r_gt < param.dmin - 2*r_gt * param.d_tol) or (2*r_gt > param.dmax + 2 * r_gt * param.d_tol))):
            
            for w in range(0,len(dt)):
                x_dt = dt[w][0]
                y_dt = dt[w][1]
                r_dt = dt[w][2]
                
                if( (gt_visit[v] == 0 and dt_visit[w] == 0) and isamatch(x_gt, y_gt, r_gt, x_dt, y_dt, r_dt, param)):
                    
                    gt_visit[v] = 1
                    dt_visit[w] = 1
                    
                    error_abs_xy = math.sqrt((x_gt-x_dt)**2 + (y_gt-y_dt)**2)
                    error_abs_r = abs(r_gt-r_dt)
                    error_rel_r = 100*(r_gt-r_dt)/r_gt
                    errors.append([r_gt, error_abs_xy, error_abs_r, error_rel_r])
                    match_ratio = 1/(abs(x_gt-x_dt) + abs(y_gt-y_dt) + 2*abs(r_gt-r_dt))
                    match_index.append([v, w, match_ratio])
                    p += 1
    
    tp_index = [i for i, e in enumerate(gt_visit) if e == 1]
    fn_index = [i for i, e in enumerate(gt_visit) if e == 0]
    fp_index = [i for i, e in enumerate(dt_visit) if e == 0]
    
    tp = len(tp_index)
    fn = len(fn_index)
    fp = len(fp_index)
    
    # global rates
    global_res_1 = np.hstack((gt[tp_index,:], np.zeros((tp,1))))
    global_res_2 = np.hstack((dt[fp_index,:], np.ones((fp,1))))
    global_res_3 = np.hstack((gt[fn_index,:], np.dot(2 , np.ones((fn,1)))))
    global_res = np.concatenate((global_res_1, global_res_2, global_res_3), axis=0)
    
    # show the original image??
    
    #theta = 0 : (2 * pi / nseg) : (2 * pi);
    theta = np.linspace(0.0, (2 * math.pi), (nseg + 1))
    
    r_tp = []
    r_fp = []
    r_fn = []
    
    if save_figs:
        implot = plt.imshow(img)
    
    for c in range(len(global_res)):
        
        x_res = global_res[c,0]
        y_res = global_res[c,1]
        r_res = global_res[c,2]
        flag_res = global_res[c,3]
        
        pline_x = np.add(np.dot(r_res , np.cos(theta)), x_res)
        pline_y = np.add(np.dot(r_res , np.sin(theta)), y_res)
        L = ""
        
        if flag_res == 0.0 :
            L = 'g'
            r_tp.append(r_res)
        
        elif flag_res == 1.0:
            L = 'r'
            r_fp.append(r_res)
        elif flag_res == 2.0:
            L = 'b'
            r_fn.append(r_res)
        else:
            print("Unknown results")
        
        # if show_figs, plot(pline_x, pline_y, strcat(L,'-'),'LineWidth',2); end
        if save_figs:
            plt.plot(pline_x, pline_y, (L + '-') , linewidth=1.0)


    if save_figs:
        # show the previous plot
        #plt.show()
        plt.savefig(save_path +'_evaluation.png', bbox_inches='tight', dpi=1000)
        plt.show()
            
        save_gt(img, gt, save_path)
            
        # plots (https://matplotlib.org/users/pyplot_tutorial.html)
        plt.figure(1)
        plt.subplot(221)
        plt.plot(np.dot(2,[item[0] for item in errors]), [item[1] for item in errors], 'bo')
        plt.title('precision of the detected position (px)')
        plt.xlabel('diameter (px)')
        plt.ylabel('error in position (px)')
            
        plt.subplot(222)
        plt.plot(np.dot(2,[item[0] for item in errors]), [item[2] for item in errors], 'bo')
        plt.title('precision of the detected radius (px)')
        plt.xlabel('diameter (px)')
        plt.ylabel('error in radius (px)')
            
        plt.subplot(223)
        plt.plot(np.dot(2,[item[0] for item in errors]), [item[3] for item in errors], 'bo')
        plt.title('precision of the detected radius (%)')
        plt.xlabel('diameter (px)')
        plt.ylabel('error in radius (%)')
        
        plt.show()
        
        totalerror_position = sum([item[1] for item in errors])
        totalerror_radius = sum([abs(x[2]) for x in errors] )
        
        # writting to text filte
        f = open(save_path +'_stats.txt','w')
        
        print("Total error in position: " + str(totalerror_position) + " , Total error in radius: " + str(totalerror_radius) )
        f.write("Total error in position: " + str(totalerror_position) + " , Total error in radius: " + str(totalerror_radius) )
        
        print("Trues, TP: " + str(tp) + " , Falses, FP: " + str(fp) + " , FN: ", str(fn) )
        f.write("Trues, TP: " + str(tp) + " , Falses, FP: " + str(fp) + " , FN: ", str(fn) )
        
        dr = tp/(tp+fn)
        fr = fp/(tp+fp)
        bf = fp/tp
        qr = tp/(tp+fp+fn)
        f_measure = 2*tp/(2*tp+fp+fn)
        
        print("f-measure: %.5f , detection percentage: %.5f , branching factor: %.5f , quality percentage: %.5f" % (f_measure, dr, bf,qr))
        f.write("f-measure: %.5f , detection percentage: %.5f , branching factor: %.5f , quality percentage: %.5f" % (f_measure, dr, bf,qr))
        f.close()
        
        if save_figs :
            plot_dacc(r_tp, r_fp, r_fn, save_path, param)
    

    return dr, fr, qr, bf, f_measure, tp, fp, fn , dt_tp, dt_fp, dt_fn


