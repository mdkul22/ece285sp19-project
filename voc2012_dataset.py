## Kinda raw, but hey, it works
##  --Darren

class VOC2012Dataset(td.Dataset):
    def __init__(self, root_dir, mode="train", image_size=(375, 500)):
        super(VOC2012Dataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        #self.data = pd.read_csv(os.path.join(root_dir, "%s.xml" % mode))
        self.annotations_dir = os.path.join(root_dir, "Annotations")
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        
        # os.listdir returns list in arbitrary order
        self.image_names = os.listdir(self.images_dir)
        self.image_names = [image.rstrip('.jpg') for image in self.image_names]
        
        
    def __len__(self):
        return len(self.image_names)

    def __repr__(self):
        return "VOC2012Dataset(mode={}, image_size={})". \
            format(self.mode, self.image_size)
    
    def __getitem__(self, idx):
        # Get file paths for image and annotation (label)
        img_path = os.path.join(self.images_dir, \
                                "%s.jpg" % self.image_names[idx])
        lbl_path = os.path.join(self.annotations_dir, \
                                "%s.xml" % self.image_names[idx])   
        #print(lbl_path)
        #print(self.image_names)
        #print(img_path)
        #print(self.images_dir)
        
        # Get objects and bounding boxes from annotations
        lbl_tree = ET.parse(lbl_path)
        objs = []
        
        for obj in lbl_tree.iter(tag='object'):
            name = obj.find('name').text
            for box in obj.iter(tag='bndbox'):
                xmax = box.find('xmax').text
                xmin = box.find('xmin').text
                ymax = box.find('ymax').text
                ymin = box.find('ymin').text
            attr = (name, xmax, xmin, ymax, ymin)
            objs.append(attr)
        
        
        # Open and normalize the image
        img = Image.open(img_path).convert('RGB')
        transform = tv.transforms.Compose([
            #tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        
        x = transform(img)
        d = objs
        return x, d

    def number_of_classes(self):
        #return self.data['class'].max() + 1
        # TODO: make more flexible
        return 20
