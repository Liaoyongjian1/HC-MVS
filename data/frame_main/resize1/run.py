# Indicate the binary directory
MVS_BIN =	"~/data/frame_main/openmvs_build/bin"



import os
import subprocess
import sys
import time


if len(sys.argv) < 3:
    print ("Usage %s image_dir output_dir" % sys.argv[0])
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

reconstruction_dir = os.path.join(output_dir, "reconstruction_global")
rematch_reconstruction_dir = os.path.join(output_dir, "reconstruction_rematch")
mvs_dir = os.path.join(output_dir, "mvs")


print ("Using input dir  : ", input_dir)
print ("     output_dir  : ", output_dir)
process_threads = 32

if not os.path.exists(output_dir):
  os.mkdir(output_dir)

if not os.path.exists(mvs_dir):
  os.mkdir(mvs_dir)


print ("1. Densify point cloud")
pDensify = subprocess.Popen( [os.path.join(MVS_BIN, "DensifyPointCloud"), 
                                  "--input-file", input_dir+"/scene.mvs", 
                                  "-w", mvs_dir, 
                                  "-o", mvs_dir+"/"+"scene_dense.mvs",
                                  "--verbosity", "2",
				  "--fusion-mode", "0",
                                  "--max-resolution", "6400",
                                  "--min-resolution","100",
                                  "--estimate-normals", "2",
                                  "--number-views", "10",
				  "--filter-point-cloud","0",
				  "--resolution-level", "1",  #resize大小
                                  "--number-views-fuse", "2", #2,越低点云越完整
                                  "--n-EstimationIters","3",  #内循环次数
                                  "--n-EstimationIters-external","4", #外循环次数
				  "--n-photo2geo","1", 	#阈值，做几次外循环就开始做几何，这个数要小于外循环次数
                                  "--ransac-probability","0.005",
                                  "--ransac-epsilon","1.4",
                                  "--ransac-cluster","7",
 				  "--ransac-min-points","40",

				  "--n-viewspread","0",         #是否做视图传播
				  "--n-opticalflow","1",        #是否做交叉一致性
				  "--n-initTriangulate","0",    #初始化标志位 1：三角初始化 0：读图初始化
				  "--n-photometric_flow","0.26", #交叉一致性权重
                                  "--n-nOptimize","1",          #帧间等滤波总标志位
                                  "--n-usepartconsistency","0", #是否做局部一致性，大于0就做
                                  "--n-usegeoconsistency","1",  #是否做几何一致性，大于0就做
				  "--use-semantic","1",         #做不做先验
                                  "--n-maxgeo_proportion","5",  #标准边放大比例  小一点好
                                  "--n-txthreshold","150",      #梯度阈值1
                                  "--n-txthreshold2","175",     #梯度阈值2
                                  "--n-para_part","0.1",        #局部一致性权重
                                  "--n-para_part2","0.05",      #局部一致性权重2
                                  "--n-para_tapa","0.26",        #几何一致性权重  0.26
                                  "--n-para_tapa2","0.26",       #几何一致性权重2 0.13
				  "--n-para_prior","0.4",         #先验代价权重
				  "--n-adapthalfwin","7",       #自适应窗口弱纹理patch一半大小
				  "--n-propagatehalfwin","5",   #传播核一半大小，最好调为5+n*propagatestep
				  "--n-propagatestep","4",      #传播核候选点选取步长


                                  "--max-threads", str(process_threads)] )
pDensify.wait()





