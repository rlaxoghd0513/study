from asyncio.windows_events import NULL
import os
import shutil
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
import time
import cv2

def load_picture(file_path):
    if not file_path.endswith(('png', 'PNG', 'jpg', 'JPG')):
        print('File ingored: ' + file_path)
        return NULL
    pic = Image.open(file_path)
    size = max(pic.width, pic.height)
    pic = pic.crop((0, 0, size, size))
    pic = pic.resize((480, 480), Image.ANTIALIAS)
    return pic        

def load_pictures(dic_path):
    file_list = os.listdir(dic_path)
    pictures = []
    for file_name in file_list:
        pic = load_picture(dic_path + file_name)
        if pic != NULL:
            pictures.append(pic)
    print(len(pictures), 'pictures loaded.')
    return pictures

class main_window:
    def __init__(self):
        # 模型路径
        self.model_path = "HRFAE-master/"
        # 输入
        self.pic_input = ""
        # 结果集
        self.pic_res = []
        # 最小年龄
        self.min_age = 20
        # 最大年龄
        self.max_age = 65
        # 年龄间隔
        self.age_sep = 1
        # 左边所有组件相对于窗口的偏移
        self.shift_y = 180
        # 窗口框架
        self.root = Tk()
        self.root.title("毕业设计: 人脸年龄编辑")
        self.root.geometry('800x600+400+300')
        self.root.minsize(800, 600)
        self.root.maxsize(800, 600)
        # 选择路径 提示文本
        self.text_input = Label(self.root, text = "输入图片: ")
        self.text_input.place(x = 25, y = self.shift_y - 100)
        # 选择路径 输入组件和值变量
        self.var_input = StringVar()
        self.entry_input = Entry (
            self.root, 
            width = 15, 
            textvariable = self.var_input
        )
        self.entry_input.place(x = 95, y = self.shift_y - 100)
        # 选择路径的按钮
        self.button_input = Button (
            self.root, 
            text = "选择本地文件", 
            command = self.select_input
        )
        self.button_input.place(x = 120, y = self.shift_y - 70)
        # 拍摄按钮
        self.button_camera_input = Button (
            self.root, 
            text = "拍摄照片", 
            command = self.camera_input
        )
        self.button_camera_input.place(x = 30, y = self.shift_y - 70)
        # 预测年龄的最小值 提示文本
        self.text_minAge = Label(self.root, text = "预测年龄最小值: ")
        self.text_minAge.place(x = 25, y = self.shift_y + 40)
        # 预测年龄的最小值 输入组件和值变量
        self.var_minAge = IntVar()
        self.spinBox_minAge = Spinbox (
            self.root,
            width = 5,
            from_ = 20,
            to = 65,  
            textvariable = self.var_minAge
        )
        self.spinBox_minAge.place(x = 130, y = self.shift_y + 40)
        # 预测年龄的最大值 提示文本
        self.text_maxAge = Label(self.root, text = "预测年龄最大值: ")
        self.text_maxAge.place(x = 25, y = self.shift_y + 80)
        # 预测年龄的最大值 输入组件和值变量
        self.var_maxAge = IntVar()
        self.spinBox_maxAge = Spinbox (
            self.root,
            width = 5,
            from_ = 20,
            to = 65, 
            textvariable = self.var_maxAge
        )
        self.spinBox_maxAge.place(x = 130, y = self.shift_y + 80)
        # 预测年龄间隔 提示文本
        self.text_ageSep = Label(self.root, text = "预测年龄间隔: ")
        self.text_ageSep.place(x = 25, y = self.shift_y + 120)
        # 预测年龄间隔 输入组件和值变量
        self.var_ageSep = IntVar()
        self.spinBox_ageSep = Spinbox (
            self.root,
            width = 5,
            from_ = 1,
            to = 10, 
            textvariable = self.var_ageSep
        )
        self.spinBox_ageSep.place(x = 130, y = self.shift_y + 120)
        # 运行模型按钮
        self.button_make = Button (
            self.root,
            text = "开始预测",
            command = self.make
        )
        self.button_make.place(x = 40, y = self.shift_y + 160)
        # 导出按钮
        self.button_export = Button (
            self.root,
            text = "导出结果",
            command = self.export_res
        )
        self.button_export.place(x = 130, y = self.shift_y + 160)
        # 比较按钮
        self.button_compare = Button (
            self.root,
            text = "比较",
            command = self.compare          
        )
        self.button_compare.place(x = 700, y = 530)
        """
        # 生成进度
        self.percent = Label(self.root)
        self.percent.place(x = 655, y = 20)
        """
        # 分隔线
        self.div = Label (
            self.root, 
            width = 0, 
            height = 33,
            bg = 'grey',
            borderwidth = 1, 
            relief="sunken"
        )
        self.div.place(x = 240, y = 20)
        # 年龄选择器
        self.age_selector = Scale (
            self.root,
            sliderlength = 20,
            from_ = 20,
            to = 65,
            resolution = 1,
            orient = HORIZONTAL,
            showvalue = False, 
            command = self.show_res,
        )
        self.age_selector.place(x = 460, y = 535)
        # 年龄选择器 上方的提示文本
        self.text_ageSelU = Label(self.root, text = "20")
        self.text_ageSelU.place(x = 500, y = 515)
        # 年龄选择器 下方的提示文本
        self.text_ageSelD = Label(self.root, text = "年龄")
        self.text_ageSelD.place(x = 495, y = 560)
        # 年龄选择器 左边的提示文本
        self.text_ageSelL = Label(self.root, text = "20")
        self.text_ageSelL.place(x = 440, y = 535)
        # 年龄选择器 右边的提示文本
        self.text_ageSelR = Label(self.root, text = "65")
        self.text_ageSelR.place(x = 565, y = 535)
        # 显示图片的图床
        self.pic_bed = Label(self.root)
        self.pic_bed.place(x = 280, y = 25)
        # 设置预测最大年龄的默认值
        self.var_maxAge.set(65)
        # 加载默认图片
        self.show_pic(load_picture("default.png"))
        # loop
        self.root.mainloop()

    # 选择图片按钮的功能
    def select_input(self):
        path = askopenfilename()
        pic = load_picture(path)
        if pic != NULL:
            self.pic_input = path
            self.var_input.set(path.split('/')[-1])
            self.show_pic(pic)
    
    # 导出结果功能
    def export_res(self):
        path = askdirectory()
        #op_export = "copy -rf {} {}".format(self.model_path + "test/output", path)
        #print(op_export)
        #os.system(op_export)
        dic_name = ''.join(('{}'.format(time.time())).split('.'))
        shutil.copytree(self.model_path + "test/output", path + '/{}'.format(dic_name))
        messagebox.showinfo(title = '提示', message = '导出成功!')

    # 在图床展示image
    def show_pic(self, ima: Image):
        pic = ImageTk.PhotoImage(ima)
        self.pic_bed.config(image = pic)
        self.pic_bed.image = pic

    # 展示当scale值为scale_val时对应的结果集的图片
    def show_res(self, scale_val):
        if int(scale_val) > self.max_age:
            id = len(self.pic_res) - 1
            cur_age = self.max_age
        else:
            id = (int(scale_val)-self.min_age) // self.age_sep
            cur_age = self.min_age + id*self.age_sep
        self.text_ageSelU.config(text = cur_age)
        if id > 0 and id < len(self.pic_res):
            self.show_pic(self.pic_res[id])
    """
    def set_percent(self, a, b):
        cur = "%d/%d"%(a, b)
        self.percent.config(text = cur)
        self.percent.text = cur
    """
    # 生成结果集(运行模型)
    def make(self):
        min_age = self.var_minAge.get()
        max_age = self.var_maxAge.get()
        age_sep = self.var_ageSep.get()
        if not self.pic_input.endswith(('png', 'PNG', 'jpg', 'JPG')):
            messagebox.showinfo(title = '错误', message = '输入格式不正确.')
            return
        if min_age > max_age:
            messagebox.showinfo(title = '错误', message = '年龄范围不正确.')
            return
        self.min_age = min_age
        self.max_age = max_age
        self.age_sep = age_sep
        self.text_ageSelL.config(text = min_age)  
        self.text_ageSelR.config(text = max_age)
        self.age_selector.config(from_ = min_age, to = max_age, resolution = age_sep)
        fmt = self.pic_input[-3:]
        op_clear_workpath1 = "rm -rf " + self.model_path + "test/input/*"
        op_clear_workpath2 = "rm -rf " + self.model_path + "test/output/*"
        os.system(op_clear_workpath1)
        os.system(op_clear_workpath2)
        shutil.copy(self.pic_input, self.model_path + "test/input/input." + fmt)
        op_run_model = "cd "+ self.model_path + " && python test.py --config 001 --min_age {} --max_age {} --age_sep {}".format(min_age, max_age, age_sep)    
        os.system(op_run_model)
        self.pic_res = load_pictures(self.model_path + "test/output/")
        self.show_res(min_age)
        messagebox.showinfo(title = '提示', message = '预测完成!')
    
    # 比较功能
    def compare(self):
        if len(self.pic_res) == 0:
            messagebox.showinfo(title = '提示', message = '请先运行模型!')
            return
        compare_window(self.min_age, self.max_age, self.age_sep)

    # 拍摄
    def camera_input(self):
        path = "test/camera_input.jpg"
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 打开摄像头
        while 1:
            ret, frame = cap.read()
            frame = cv2.flip(frame[0:480, 0:480], 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
            cv2.imshow("capture", frame)  # 生成摄像头窗口
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
                cv2.imwrite(path, frame)  # 保存路径
                break
        cap.release()
        cv2.destroyAllWindows()
        pic = load_picture(path)
        if pic != NULL:
            self.pic_input = path
            self.var_input.set(path.split('/')[-1])
            self.show_pic(pic)

class compare_window:
    def __init__(self, min_age, max_age, age_sep):
        # 模型路径
        self.model_path = "HRFAE-master/"
        # 读取图片
        self.pic_list = load_pictures(self.model_path + "test/output/")
        self.pic_list.insert(0, load_picture(self.model_path + "test/input/input.jpg"))
        # 获取当前生成的目标年龄列表
        self.target_age_list = list(range(min_age, max_age+1, age_sep))
        if (max_age - min_age) % age_sep != 0:
            self.target_age_list.append(max_age)
        self.pic_text = []
        self.pic_text.append('原始图像')
        for age in self.target_age_list: # 如果显示的图像太多, 这里改成从中平均选取9张
            self.pic_text.append('  ' + str(age) + '岁')       
        # 窗口框架
        self.root = Toplevel()
        self.root.title("比较")
        self.root.geometry('995x470+100+100')
        self.root.minsize(995, 470)
        self.root.maxsize(995, 470)
        # 图片序列    
        for i in range(10):
            j = i // 5
            text_list = Label (
                self.root, 
                font = ("黑体", 14, "bold"),
                text = self.pic_text[i]
            )
            text_list.place(x = 60 + (i-5*j)*195, y = 210+230*j)
            res_list = Label(self.root)
            res_list.place(x = 5 + (i-5*j)*195, y = 5+230*j)
            ima = self.pic_list[i].resize((200, 200), Image.ANTIALIAS)
            pic = ImageTk.PhotoImage(ima)
            res_list.config(image = pic)
            res_list.image = pic

main_window()


