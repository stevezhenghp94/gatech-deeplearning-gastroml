import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(
        Image.open("../data/demo_imgs/1.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 1 CORRECT: Shrimp pasta with bread")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open("../data/demo_imgs/2.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Scrambled eggs, toast and avocado")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(
        Image.open("../data/demo_imgs/3.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 3 CORRECT: Fried eggplant with parmesan cheese")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open("../data/demo_imgs/4.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: Pasta with ground beef and tomato sauce")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open("../data/demo_imgs/5.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: Pancakes with blueberries and strawberries")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    test_img6 = transform(
        Image.open("../data/demo_imgs/6.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 6 CORRECT: Chicken enchiladas")
    print(
        "Example 6 OUTPUT: "
        + " ".join(model.caption_image(test_img6.to(device), dataset.vocab))
    )
    test_img7 = transform(
        Image.open(
            "../Kaggle data/Food Images/Food Images/3-ingredient-caramel-apple-hand-pies.jpg"
        ).convert("RGB")
    ).unsqueeze(0)
    print("Example 7 CORRECT: Caramel apple hand pies")
    print(
        "Example 7 OUTPUT: "
        + " ".join(model.caption_image(test_img7.to(device), dataset.vocab))
    )
    test_img8 = transform(
        Image.open(
            "../Kaggle data/Food Images/Food Images/watermelon-with-yogurt-poppy-seeds-and-fried-rosemary.jpg"
        ).convert("RGB")
    ).unsqueeze(0)
    print("Example 8 CORRECT: Watermelon with yogurt, poppy seeds and fried rosemary")
    print(
        "Example 8 OUTPUT: "
        + " ".join(model.caption_image(test_img8.to(device), dataset.vocab))
    )
    test_img9 = transform(
        Image.open(
            "../Kaggle data/Food Images/Food Images/texas-style-barbecued-brisket-242249.jpg"
        ).convert("RGB")
    ).unsqueeze(0)
    print("Example 9 CORRECT: Texas style barbecued brisket")
    print(
        "Example 9 OUTPUT: "
        + " ".join(model.caption_image(test_img9.to(device), dataset.vocab))
    )
    test_img10 = transform(
        Image.open(
            "../Kaggle data/Food Images/Food Images/berry-explosion-muffins.jpg"
        ).convert("RGB")
    ).unsqueeze(0)
    print("Example 10 CORRECT: Berry explosion muffins")
    print(
        "Example 10 OUTPUT: "
        + " ".join(model.caption_image(test_img10.to(device), dataset.vocab))
    )
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
