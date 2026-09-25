import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBrain(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
     
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)


        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        
        return output

sample_input = torch.randn(1, 3, 4) 

my_ai_brain = SelfAttentionBrain(embed_dim=4)

processed_output = my_ai_brain(sample_input)

print("🧠 AI 뇌 내부 연산 완료!")
print("입력 데이터 크기:", sample_input.shape)
print("문맥 처리 후 출력 크기:", processed_output.shape)
