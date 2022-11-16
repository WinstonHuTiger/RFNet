import os
import shutil
source_path = r"/home/qingqiao/RFNet/data/fft_data_processed"

envs = os.listdir(source_path)
env1_people_num = 20
env2_people_num = 4
acts = ['jm', 'run', 'sit', 'squat', 'walk', 'rip', 'throw', 'wip']
for env in envs:
    p_ = os.path.join(source_path, env)
    print("The envrionment", env)
    #acts = os.listdir(p_)

    for act in acts:
        third_path = os.path.join(p_, act)
        person_images = os.listdir(third_path)
        if env == 'env1':
            person_int = len(person_images)//env1_people_num
        else:
            person_int = len(person_images)//env2_people_num
        for image in person_images:
            file_name, _ = image.split(".")
            file_int = int(file_name)
            person = None
            if file_int % person_int !=0:
                if file_int // person_int +1 > 9:
                    person = "p" + str(file_int // person_int + 1)
                else:
                    person ="p0" + str(file_int// person_int + 1)                   
            else:
                if file_int // person_int > 9:
                    person = "p" + str(file_int // person_int)
                else:
                    person = "p0" + str(file_int // person_int)
            person_path = os.path.join(p_, person)
            if not os.path.isdir(person_path):
                os.mkdir(person_path)
            
            person_act_path = os.path.join(p_, person, act)
            if not os.path.isdir(person_act_path):
                os.mkdir(person_act_path)
            
            shutil.copyfile(os.path.join(p_, act, image), os.path.join(person_act_path, image))