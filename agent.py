import os
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class AgentTools:
    def __init__(self, workspace_path):
        self.workspace = os.path.abspath(workspace_path)
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

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
        return f"✅ {filename} 파일이 성공적으로 저장되었습니다."

    def run_terminal(self, command):
        dangerous_commands = ["rm -rf /", "rm -rf /*", "shutdown", "reboot"]
        for cmd in dangerous_commands:
            if cmd in command:
                return f"❌ [보안 거부] '{command}' 명령어는 안전을 위해 금지되었습니다."

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return f"✅ [터미널 실행 성공]:\n{result.stdout}"
            else:
                return f"⚠️ [터미널 실행 에러]:\n{result.stderr}"
        except Exception as e:
            return f"❌ [실행 예외]: {str(e)}"

class AIAgent:
    def __init__(self, workspace="./my_workspace"):
        self.tools = AgentTools(workspace)
        self.brain = SelfAttentionBrain(embed_dim=4)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.01)

    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.brain(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run_task(self, filename, code_content):
        print("🧠 [2번] AI 뇌가 판단 중...")
        sample_input = torch.randn(1, 3, 4)
        brain_decision = self.brain(sample_input)
        print(f"   -> 뇌 연산 출력 크기: {brain_decision.shape}")

        print("\n💾 [3번] 파일 생성/수정 도구 실행 중...")
        write_res = self.tools.write_file(filename, code_content)
        print(f"   -> {write_res}")

        print("\n💻 [3번] 터미널 검증 도구 실행 중...")
        term_res = self.tools.run_terminal(f"python3 {filename}")
        print(f"   -> {term_res}")

if __name__ == "__main__":
    agent = AIAgent()

    dummy_input = torch.randn(1, 3, 4)
    target = torch.zeros(1, 3, 4)
    loss_val = agent.train_step(dummy_input, target)
    print(f"📉 [1번 경사하강법+최적화] 오차(Loss) 측정값: {loss_val:.4f}\n")

    test_code = "print('단일 파이썬 파일로 AI 에이전트 실행 성공!')"
    agent.run_task("test_script.py", test_code)
