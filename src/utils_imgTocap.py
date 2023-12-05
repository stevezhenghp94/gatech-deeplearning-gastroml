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
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
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
