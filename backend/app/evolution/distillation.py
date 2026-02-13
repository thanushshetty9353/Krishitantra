import torch
import torch.nn.functional as F


def distill(teacher, student, tokenizer, steps=200, device="cpu"):
    teacher.to(device)
    student.to(device)

    teacher.eval()
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)

    prompts = [
        "Explain quantum computing.",
        "What is artificial intelligence?",
        "Translate hello to French.",
        "Summarize climate change.",
        "Benefits of exercise."
    ]

    for step in range(steps):
        prompt = prompts[step % len(prompts)]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            teacher_outputs = teacher(**inputs, labels=inputs["input_ids"])
            teacher_logits = teacher_outputs.logits

        student_outputs = student(**inputs, labels=inputs["input_ids"])
        student_logits = student_outputs.logits

        loss = F.mse_loss(student_logits, teacher_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student.eval()
    return student
