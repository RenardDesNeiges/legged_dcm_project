#In the following we import the classes that we need for simulation
import time
import pybullet #pybullet simulator
import pybullet_data
import numpy as np # numpy library for matrix computatios
from FootTrajectoryGenerator import * # Foot trajectory generation Class
#You sould uncomment the following line after developing DCM part
from DCMTrajectoryGenerator import * #  DCM trajectory generation Class(will be implemented by students)
from RobotUtils import * # Class related to Inverse Kinematics 

def walk_run(params):

    GRAVITY = 9.81
    cost_of_transport_list = []

    
    """
    ============================================================
        Setting up environment
    ============================================================
    """

    #In the following we create an object of dynamic engine of pybullet and use direct simulation (no GUI)
    if params["GUI"]:
        phisycsClient = pybullet.connect(pybullet.GUI) 
    else:
        phisycsClient = pybullet.connect(pybullet.DIRECT) 
        
        
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    #In the following we load the urdf model of the robot and we specify the setting for the simulation
    pybullet.resetSimulation()
    planeID = pybullet.loadURDF("plane.urdf")
    if params["GUI"]:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
    pybullet.setGravity(0,0,-GRAVITY)
    atlas=robotID = pybullet.loadURDF("atlas/atlas_v4_with_multisense.urdf", [0,0,0.93],useFixedBase = 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
    pybullet.setRealTimeSimulation(0)

    """
    ============================================================
        Planning feet trajectories
    ============================================================
    """
    
    #In this part we will specify the steps position and duration and we will implement foot trajectory generation
    doubleSupportDuration = params["doubleSupportDuration"] #We select 0.25 second as DS duration
    stepDuration = params["stepDuration"] #We select 1.2 second as step duration(step duration=SS+DS)
    pelvisHeight= params["pelvisHeight"] #Constant pelvis(CoM) height during walking
    maximumFootHeight = params["maximumFootHeight"] #The maximum height of swing foot during each step

    FootPlanner = FootTrajectoryGenerator(stepDuration, doubleSupportDuration, maximumFootHeight) #We create an object of foot FootTrajectoryGenerator Class
    stepWidth=params["stepWidth"] #=lateralDistanceOfFeet/2
    stepLength=params["stepLength"] #longitudinal distance between two sequential feet stepLength=stepStride/2, 
    numberOfFootPrints=params["numberOfFootPrints"]
    FootPrints=np.empty((numberOfFootPrints, 3))

    #In the following we define the foot step positions(Ankle joint position projected on foot print)
    for i in range(0,numberOfFootPrints):
        if(i%2==0):
            if(i==0):
                FootPrints[i][:]=[i*stepLength,stepWidth,0.0]
            elif(i==numberOfFootPrints-1):
                FootPrints[i][:]=[(i-2)*stepLength,stepWidth,0.0]            
            else:
                FootPrints[i][:]=[(i-1)*stepLength,stepWidth,0.0]
        else:
            FootPrints[i][:]=[(i-1)*stepLength,-stepWidth,0.0]
                

    FootPlanner.setFootPrints(FootPrints)#We set the foot step positions
    FootPlanner.generateTrajectory() #We generate the foot trajectory 
    leftFootTrajectory = np.array(FootPlanner.getLeftFootTrajectory())
    rightFootTrajectory = np.array(FootPlanner.getRightFootTrajectory())

    """
    ============================================================
        Planning CoM and DCM trajectories
    ============================================================
    """

    CoPOffset=np.array([0.03,0.00]) #Offset between CoP and footprint position(Ankle position) 

    DCMPlanner = DCMTrajectoryGenerator(pelvisHeight, stepDuration, doubleSupportDuration)#We create an object of foot DCMTrajectoryGenerator Class

    CoPPositions=np.empty((DCMPlanner.numberOfSteps+1, 3))#Initialization of the CoP array

    for i in range(0,DCMPlanner.numberOfSteps+1):
        if(i%2!=0):
            CoPPositions[i][:]=[(i)*stepLength-CoPOffset[0],stepWidth-CoPOffset[1],0.0]
            if(i==1):
                CoPPositions[i][:]=[(i)*stepLength,stepWidth-CoPOffset[1],0.0]
        else:
            CoPPositions[i][:]=[(i)*stepLength-CoPOffset[0],-stepWidth+CoPOffset[1],0.0]
            if(i==0):
                CoPPositions[i][:]=[(i)*stepLength,-stepWidth+CoPOffset[1],0.0]

    DCMPlanner.setCoP(CoPPositions)#We set the desired CoP positions

    DCMPlanner.setFootPrints(FootPrints)#We set the foot steps positions

    DCMTrajectory = DCMPlanner.getDCMTrajectory()#At the end of DCM Planning the size of DCM vector should be 4320


    CoPOffset=np.array([-0.0,0.016]) #Offset between CoP and footprint position(Ankle position) 

    DCMPlanner = DCMTrajectoryGenerator(pelvisHeight, stepDuration, doubleSupportDuration)#We create an object of foot DCMTrajectoryGenerator Class
    CoPPositions=np.empty((DCMPlanner.numberOfSteps+1, 3))#Initialization of the CoP array

    #In the following we define the foot step positions
    for i in range(0,DCMPlanner.numberOfSteps+1):
        if(i%2!=0):
            CoPPositions[i][:]=[(i)*stepLength-CoPOffset[0],stepWidth-CoPOffset[1],0.0]
            if(i==1):
                CoPPositions[i][:]=[(i)*stepLength,stepWidth-CoPOffset[1],0.0]
        else:
            CoPPositions[i][:]=[(i)*stepLength-CoPOffset[0],-stepWidth+CoPOffset[1],0.0]
            if(i==0):
                CoPPositions[i][:]=[(i)*stepLength,-stepWidth+CoPOffset[1],0.0]
            
    DCMPlanner.setCoP(CoPPositions)#We set the desired CoP positions
    DCMPlanner.setFootPrints(FootPrints)#We set the foot steps positions
    DCMTrajectory = DCMPlanner.getDCMTrajectory()#At the end of DCM Planning the size of DCM vector should be 4320
    initialCoM = np.array([0.0,0.0,DCMPlanner.CoMHeight])
    comTrajectory = DCMPlanner.getCoMTrajectory(initialCoM)
    DCMPlanner.calculateCoPTrajectory()

    DCMPlanner.alpha = 0.5;

    """
    ============================================================
        Implementing trajectories on the robot
    ============================================================
    """

    AtlasUtils = RobotUtils()#This object is responsible for inverse kinematics

    #Preparing the constant joints position for the upper-body joints
    leftArmPositions=[-0.2,-1.7,0.3,-0.3,0.0,0.0,0.0]
    rightArmPositions=[ 0.2, 1.7,0.3, 0.3,0.0,0.0,0.0]
    bodyPositions = [0.0,0.0,0.0,0.0]
    leftArmIndex = [3,4,5,6,7,8,9]
    rightArmIndex= [11,12,13,14,15,16,17]
    bodyIndex = [0,1,2,10]
    leftLegIndex=[18,19,20,21,22,23]
    rightLegIndex=[24,25,26,27,28,29]
    jointsIndex= np.zeros(30)
    jointsPositions= np.zeros(30)
    jointsIndex[0:4]=bodyIndex
    jointsIndex[4:11]=rightArmIndex
    jointsIndex[11:18]=leftArmIndex
    jointsIndex[18:24]=rightLegIndex
    jointsIndex[24:30]=leftLegIndex
    jointsPositions[0:4]=bodyPositions
    jointsPositions[4:11]=rightArmPositions
    jointsPositions[11:18]=leftArmPositions


    for i in range(1000):  #1000 sampling time will be allocated for the initialization of the robot position
        lowerBodyJoints= AtlasUtils.doInverseKinematics([0.0,0.0,0.9 - (i/1000)* (0.9-DCMPlanner.CoMHeight)], np.eye(3),[0.0,0.13,0.0], np.eye(3),[0.0, -0.13,0.0], np.eye(3))
        jointsPositions[24:30] = lowerBodyJoints[6:12]
        jointsPositions[18:24] = lowerBodyJoints[0:6]
        pybullet.setJointMotorControlArray(bodyIndex=robotID,
                                    jointIndices=jointsIndex,
                                    controlMode=pybullet.POSITION_CONTROL,
                                    targetPositions = jointsPositions)
        pybullet.stepSimulation()
        time.sleep(0.0002)
        
    time.sleep(1)#Just wait for a second
    
    velocities = []
    positions = []

    #Solving the inverse kinematic and sending the joints position command to the robot
    for i in range(int((DCMPlanner.numberOfSamplesPerSecond) * CoPPositions.shape[0] * DCMPlanner.stepDuration)):
        lowerBodyJoints = AtlasUtils.doInverseKinematics(comTrajectory[i], np.eye(3),leftFootTrajectory[i], np.eye(3),rightFootTrajectory[i], np.eye(3))
        jointsPositions[24:30] = lowerBodyJoints[6:12]
        jointsPositions[18:24] = lowerBodyJoints[0:6]
        pybullet.setJointMotorControlArray(bodyIndex=robotID,
                                    jointIndices=jointsIndex,
                                    controlMode=pybullet.POSITION_CONTROL,
                                    targetPositions = jointsPositions)
        
        
        
        #The following part, at every 14 sampling period, will plot the foot and pelvis trajectory in pybullet 
        if(i>14 and i%14==0):
            pybullet.addUserDebugLine(comTrajectory[i-14],comTrajectory[i],[0,0.9,0.0],4,140)
            pybullet.addUserDebugLine(leftFootTrajectory[i-14],leftFootTrajectory[i],[1.0,0.0,0.8],4,140)
            pybullet.addUserDebugLine(rightFootTrajectory[i-14],rightFootTrajectory[i],[0.1,0.9,0.8],4,140)
        
        velocities.append(pybullet.getBaseVelocity(bodyUniqueId=robotID))
        positions.append(pybullet.getBasePositionAndOrientation(bodyUniqueId=robotID))
        
        # simulation step
        pybullet.stepSimulation()
        
    current_motor_torques = np.array([x[3] for x in pybullet.getJointStates(bodyUniqueId=robotID, jointIndices=jointsIndex)])
    current_motor_velocities = np.array([x[1] for x in pybullet.getJointStates(bodyUniqueId=robotID, jointIndices=jointsIndex)])

    current_base_velocitites = np.array([x[0] for x in pybullet.getBaseVelocity(bodyUniqueId=robotID)])[0] # get linear x vel
    current_mass = np.array([x for x in pybullet.getDynamicsInfo(bodyUniqueId=robotID, linkIndex=-1)])[0] # get base weight
    # HOW TO GET OTHER WEIGHTS????

    current_power = np.dot(current_motor_torques.T, current_motor_velocities)
    current_cost_of_transport = current_power / (current_mass * GRAVITY * current_base_velocitites)
    cost_of_transport_list.append(current_cost_of_transport)    
        
    for i in range(1,1000):
        velocities.append(pybullet.getBaseVelocity(bodyUniqueId=robotID))
        positions.append(pybullet.getBasePositionAndOrientation(bodyUniqueId=robotID))
        pybullet.stepSimulation()
        
    pybullet.disconnect()
    
    cost_of_transport = np.sum(np.array(cost_of_transport_list))
        
    vel = np.array(velocities)
    pos = np.array([*np.array(positions)[:,0]])
    delta_time = CoPPositions.shape[0] * DCMPlanner.stepDuration
    
    filename = "vel_pos_" + str(stepDuration) + "_" + str(stepLength) + ".npy"
    filepath = "logs/" + filename
    
    with open(filepath, 'wb') as f:
        np.save(f, velocities)
        np.save(f, positions)
        
    print("Successfull sim !")

    return  vel, pos, delta_time, cost_of_transport
"""
============================================================
main
============================================================
"""



# #In this part we will specify the steps position and duration and we will implement foot trajectory generation
# params = {
#     "doubleSupportDuration" : 0.25,
#     "stepDuration" : 1.2,
#     "pelvisHeight" : 0.7,
#     "maximumFootHeight" : 0.07,
#     "stepWidth" :0.12,
#     "stepLength" :0.1,
#     "numberOfFootPrints" :17,
# }

# walk_run(params)