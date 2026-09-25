import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. AI 두뇌 (신경망 모델) - PyTorch 최적화
# ==========================================
class SelfAttentionBrain(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # PyTorch nn.Linear를 통한 가중치 및 기울기 자동 관리
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # 1. Linear + GELU 활성화 함수 적용 (C++/CUDA 벡터화 연산)
        q = F.gelu(self.q_proj(x), approximate="tanh")
        k = F.gelu(self.k_proj(x), approximate="tanh")
        v = F.gelu(self.v_proj(x), approximate="tanh")

        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output

# ==========================================
# 2. 에이전트 도구 (파일 입출력 및 터미널)
# ==========================================
class AgentTools:
    def __init__(self, workspace_path):
        self.workspace = os.path.abspath(workspace_path)
        os.makedirs(self.workspace, exist_ok=True)

    def read_file(self, filename):
        filepath = os.path.join(self.workspace, filename)
        if not os.path.exists(filepath):
            return f"❌ [에러] {filename} 파일이 없습니다."
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, filename, content):
        filepath = os.path.join(self.workspace, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ {filename} 파일이 저장되었습니다."

    def run_terminal(self, command):
        dangerous_commands = ["rm -rf /", "shutdown", "reboot"]
        if any(cmd in command for cmd in dangerous_commands):
            return f"❌ [보안 거부] '{command}' 금지됨."

        try:
            res = subprocess.run(
                command, shell=True, cwd=self.workspace, 
                capture_output=True, text=True, timeout=10
            )
            return f"✅ [실행 성공]:\n{res.stdout.strip()}" if res.returncode == 0 else f"⚠️ [실행 에러]:\n{res.stderr.strip()}"
        except Exception as e:
            return f"❌ [예외]: {str(e)}"

# ==========================================
# 3. 자율 AI 에이전트 루프
# ==========================================
class AIAgent:
    def __init__(self, workspace="./my_workspace"):
        # 디바이스 자동 감지 (GPU/MPS/CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tools = AgentTools(workspace)
        self.brain = SelfAttentionBrain(embed_dim=4).to(self.device)
        
        # 옵티마이저 및 손실 함수 설정
        self.optimizer = torch.optim.SGD(self.brain.parameters(), lr=0.05)
        self.criterion = nn.MSELoss()

    def train_step(self, inputs, targets):
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)

        # 역전파 학습 파이프라인
        self.optimizer.zero_grad()
        outputs = self.brain(inputs_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()  # 자동 미분 적용
        self.optimizer.step()

        return loss.item()

    def run_task(self, filename, code_content):
        device_name = str(self.device).upper()
        print(f"🧠 [AI 두뇌 연산] 장치 ({device_name}) 가속 실행 중...")
        
        # 추론 모드 (메모리 절약)
        self.brain.eval()
        with torch.no_grad():
            sample_input = torch.tensor([[0.5, -0.2, 0.1, 0.9], [0.1, 0.8, -0.4, 0.3]], 
                                        dtype=torch.float32, device=self.device)
            decision = self.brain(sample_input)
        print(f"   -> 신경망 연산 결과:\n{decision.cpu().numpy()}\n")
        self.brain.train()

        print("💾 파일 생성 및 저장 중...")
        write_res = self.tools.write_file(filename, code_content)
        print(f"   -> {write_res}\n")

        print("💻 터미널 자율 검증 중...")
        term_res = self.tools.run_terminal(f"python3 {filename}")
        print(f"   -> {term_res}")

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    agent = AIAgent()

    # 1. 역전파 학습 테스트
    dummy_input = [[0.1, 0.2, 0.3, 0.4]]
    target = [[0.0, 0.0, 0.0, 0.0]]

    print("📉 [자동 미분 기반 고속 역전파 학습 진행]")
    for step in range(1, 4):
        loss_val = agent.train_step(dummy_input, target)
        print(f"   Step {step} - Loss: {loss_val:.8f}")

    print("\n" + "-"*50 + "\n")

    # 2. 에이전트 자율 작업 테스트
    test_code = "print('최적화된 AI 에이전트 파이프라인 정상 동작 완료!')"
    agent.run_task("test_script.py", test_code)
