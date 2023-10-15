import torch
from sqapi.helpers import create_parser
from sqapi.annotate import Annotator
from sqapi.request import query_filter as qf
from PIL import Image
from torchvision import transforms
from prediction_model import KelpClassifier
from torch.nn import functional as F
import time

class TorchBOT(Annotator):

    def __init__(self, crop_perc, model_path: str = 'models/Ecklonia/epoch=49-step=32550.ckpt', 
                **kwargs: object) -> object:
        """
        Uses pytorch to run a pytorch model
        :param model_path: the path of the pytorch model
        :param crop_perc: defines the patch size
        :param network: the network to use for the model
        """

        # Instantiate the Trainer with specific settings
        acc_val = 'cpu'
        if torch.cuda.is_available(): acc_val = 'gpu'

        super().__init__(**kwargs)
        self.model = KelpClassifier.load_from_checkpoint(
            model_path,
            optimizer="AdamW",
            backbone_name='inception_v3',
            map_location=torch.device(acc_val)
        )
        
        self.model.eval()
        self.crop_perc = crop_perc

    def predict(self, input_data):
        # Convert the input_data to tensor directly without DataLoader
        train_transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),  # Convert image to tensor [0, 255] -> [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = train_transforms(input_data).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            y_hat = self.model(img_tensor)
            prob = F.softmax(y_hat, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            classifier_code = int(top_class.data[0][0])
            prob = float(top_p.data[0][0])
            
        return classifier_code, prob

    def get_patch(self, x, y, mediaobj):
        # check if data has been downloaded and if not download it
        if not mediaobj.is_processed:
            orig_image = mediaobj.data()
            img = mediaobj.data(Image.fromarray(orig_image))
        else:
            img = mediaobj.data()   # has already been padded, so will return padded image

        # Calculate the crop percentage and size around the specified x-y point
        crop_size = ((img.size[0] + img.size[1]) / 2) * self.crop_perc
        x = img.size[0] * x  # Center position
        y = img.size[1] * y  # Center position
        x0, x1, y0, y1 = self.get_crop_points(x, y, img, crop_size)

        # Crop the image to the specified region of interest
        cropped_img = img.crop((x0, y0, x1, y1))

        return cropped_img

    def get_crop_points(self, x, y, original_image, img_size):
        # Calculate the crop points for the specified x-y position to extract a region of interest around the point
        x_img, y_img = original_image.size
        crop_dist = img_size / 2
        if x - crop_dist < 0: x0 = 0
        else: x0 = x - crop_dist

        if y - crop_dist < 0: y0 = 0
        else: y0 = y - crop_dist

        if x + crop_dist > x_img: x1 = x_img
        else: x1 = x + crop_dist

        if y + crop_dist > y_img: y1 = y_img
        else: y1 = y + crop_dist

        return int(x0), int(x1), int(y0), int(y1)
    
    def classify_point(self, mediaobj, x, y, t):
        """ returns: classifier_code, prob """
        patch_img = self.get_patch(x, y, mediaobj)
        classifier_code, prob = self.predict(patch_img)

        return classifier_code, float(prob)

if __name__ == '__main__':

    # Running `bot = cli_init(RandoBOT)` would normally do all the steps below and initialise the class,
    # but in this instance we cant to add some extra commandline arguments to decide what annotation_sets to process

    # Get the cli arguments from the Class __init__ function signatures
    parser = create_parser(TorchBOT)

    # Add some additional custom cli args not related to the model
    parser.add_argument('--annotation_set_id', help="Process specific annotation_set", type=int, default = 10742) #8322
    parser.add_argument('--user_group_id', help="Process all annotation_sets contained in a specific user_group", type=int)
    parser.add_argument('--affiliation_group_id', help="Process all annotation_sets contained in a specific Affiliation", type=int)
    parser.add_argument('--after_date', help="Process all annotation_sets after a date YYYY-MM-DD", type=str)
    parser.add_argument('--media_count_max', help="Filter annotation_sets that have less than a specific number of media objects", type=int)
    parser.add_argument('--crop_perc', help="Which crop percentage to use for the image patch as float.", type=float, default=0.18)
    parser.add_argument('--label_map_file', help="Path to the label map file.", type=str, default='bots/kelp_bot_label_map.json')
    parser.add_argument('--host', help="Host to run the bot on.", type=str, default='https://squidle.org')
    args = parser.parse_args()
    # Set the host, API key, and label map file for the bot
    #open text file in read mode
    text_file = open("bots/API_KEY.txt", "r")
    #read whole file to a string
    api_key = text_file.read()
    #close file
    text_file.close()
    args.api_key = api_key

    bot = TorchBOT(**vars(args))

    # Initialise annotation_set request using sqapi instance in Annotator class
    r = bot.sqapi.get("/api/annotation_set")

    # Only return annotation_sets that do not already have suggestions from this user
    r.filter_not(qf("children", "any", val=qf("user_id", "eq", bot.sqapi.current_user.get("id"))))

    # Filter annotation sets based on ID
    if args.annotation_set_id:
        r.filter("id", "eq", args.annotation_set_id)

    # Constrain date ranges to annotation_sets ceated after a specific date
    if args.after_date:
        r.filter("created_at", "gt", args.after_date)

    # Filter annotation_sets based on a user group
    if args.user_group_id:
        r.filter("usergroups", "any", val=qf("id", "eq", args.user_group_id))

    if args.affiliation_group_id:
        r.filter("user", "has", val=qf("affiliations_usergroups", "any", val=qf("group_id", "eq", args.affiliation_group_id)))

    if args.media_count_max:
        r.filter("media_count", "lte", args.media_count_max)

    # Start the bot in a loop that polls at a defined interval
    bot.start(r)
