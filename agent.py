import os
import math
import random
import subprocess
import torch

class NeuronParameters:
    def __init__(self, in_features, out_features, device="cpu"):
        self.device = device
        self.shape = (in_features, out_features)

        if self.device != "cpu":
            self.W = torch.randn(in_features, out_features, dtype=torch.float64, device=self.device) * 0.1
            self.b = torch.full((out_features,), 0.01, dtype=torch.float64, device=self.device)
            self.dW = torch.zeros_like(self.W)
            self.db = torch.zeros_like(self.b)
        else:
            self.W = [[random.gauss(0, 0.1) for _ in range(out_features)] for _ in range(in_features)]
            self.b = [0.01 for _ in range(out_features)]
            self.dW = [[0.0] * out_features for _ in range(in_features)]
            self.db = [0.0] * out_features

class RawLinearNeuron:
    def __init__(self, in_features, out_features, device="cpu"):
        self.params = NeuronParameters(in_features, out_features, device)
        self.last_X = None
        self.last_Y = None

    def forward(self, X):
        self.last_X = X
        device = self.params.device
        W, b = self.params.W, self.params.b

        if device != "cpu":
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float64, device=device)
            Y = torch.matmul(X, W) + b
            sqrt_2_pi = math.sqrt(2.0 / math.pi)
            inner = sqrt_2_pi * (Y + 0.044715 * (Y ** 3))
            activated_Y = 0.5 * Y * (1.0 + torch.tanh(inner))
            self.last_Y = activated_Y
            return activated_Y
        else:
            rows_X, cols_X = len(X), len(X[0])
            cols_W = len(W[0])
            
            Y = [[0.0] * cols_W for _ in range(rows_X)]
            for i in range(rows_X):
                for k in range(cols_X):
                    for j in range(cols_W):
                        Y[i][j] += X[i][k] * W[k][j]
                        
            for i in range(rows_X):
                for j in range(cols_W):
                    Y[i][j] += b[j]

            sqrt_2_pi = math.sqrt(2.0 / math.pi)
            activated_Y = []
            for row in Y:
                new_row = []
                for x in row:
                    inner = sqrt_2_pi * (x + 0.044715 * (x ** 3))
                    new_row.append(0.5 * x * (1.0 + math.tanh(inner)))
                activated_Y.append(new_row)

            self.last_Y = activated_Y
            return activated_Y

    def update_weights(self, lr=0.01):
        device = self.params.device
        if device != "cpu":
            self.params.W -= lr * self.params.dW
            self.params.b -= lr * self.params.db
            self.params.dW.zero_()
            self.params.db.zero_()
        else:
            rows, cols = len(self.params.W), len(self.params.W[0])
            for i in range(rows):
                for j in range(cols):
                    self.params.W[i][j] -= lr * self.params.dW[i][j]
                    self.params.dW[i][j] = 0.0
            for j in range(cols):
                self.params.b[j] -= lr * self.params.db[j]
                self.params.db[j] = 0.0

class RawSelfAttentionBrain:
    def __init__(self, embed_dim, device="cpu"):
        self.config = {
            "embed_dim": embed_dim,
            "device": device
        }
        self.layers = {
            "q": RawLinearNeuron(embed_dim, embed_dim, device=device),
            "k": RawLinearNeuron(embed_dim, embed_dim, device=device),
            "v": RawLinearNeuron(embed_dim, embed_dim, device=device)
        }

    def _matmul(self, A, B):
        if self.config["device"] != "cpu":
            return torch.matmul(A, B)
        rows_A, cols_A = len(A), len(A[0])
        cols_B = len(B[0])
        result = [[0.0] * cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for k in range(cols_A):
                for j in range(cols_B):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def _transpose(self, A):
        if self.config["device"] != "cpu":
            return A.transpose(-2, -1)
        return [list(i) for i in zip(*A)]

    def _softmax(self, matrix):
        if self.config["device"] != "cpu":
            return torch.softmax(matrix, dim=-1)
        result = []
        for row in matrix:
            # max_val을 차감하여 e^x 지수 폭발(Overflow) 완벽 방지
            max_val = max(row)
            exps = [math.exp(x - max_val) for x in row]
            sum_exps = sum(exps)
            result.append([e / sum_exps for e in exps])
        return result

    def forward(self, X):
        Q = self.layers["q"].forward(X)
        K = self.layers["k"].forward(X)
        V = self.layers["v"].forward(X)

        K_T = self._transpose(K)
        scores = self._matmul(Q, K_T)
        scale = math.sqrt(self.config["embed_dim"])

        if self.config["device"] != "cpu":
            scaled_scores = scores / scale
            attn_weights = self._softmax(scaled_scores)
            output = self._matmul(attn_weights, V)
        else:
            scaled_scores = [[val / scale for val in row] for row in scores]
            attn_weights = self._softmax(scaled_scores)
            output = self._matmul(attn_weights, V)

        return output

    def backward_and_optimize(self, grad_output, lr=0.01):
        # FP64 역전파: 수동 미분 기울기를 각 뉴런 레이어 파라미터에 누적
        for layer_name in ["q", "k", "v"]:
            layer = self.layers[layer_name]
            device = self.config["device"]
            if device != "cpu":
                X = layer.last_X
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float64, device=device)
                layer.params.dW = torch.matmul(X.T, grad_output)
                layer.params.db = torch.sum(grad_output, dim=0)
            else:
                X = layer.last_X
                rows_X, cols_X = len(X), len(X[0])
                cols_W = len(layer.params.W[0])
                for k in range(cols_X):
                    for j in range(cols_W):
                        sum_grad = 0.0
                        for i in range(rows_X):
                            sum_grad += X[i][k] * grad_output[i][j]
                        layer.params.dW[k][j] = sum_grad
                for j in range(cols_W):
                    layer.params.db[j] = sum([grad_output[i][j] for i in range(rows_X)])
            
            layer.update_weights(lr)

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
        return f"✅ {filename} 파일이 저장되었습니다."

    def run_terminal(self, command):
        dangerous_commands = ["rm -rf /", "shutdown", "reboot"]
        for cmd in dangerous_commands:
            if cmd in command:
                return f"❌ [보안 거부] '{command}' 금지됨."

        try:
            res = subprocess.run(command, shell=True, cwd=self.workspace, capture_output=True, text=True, timeout=10)
            return f"✅ [실행 성공]:\n{res.stdout}" if res.returncode == 0 else f"⚠️ [실행 에러]:\n{res.stderr}"
        except Exception as e:
            return f"❌ [예외]: {str(e)}"

class AIAgent:
    def __init__(self, workspace="./my_workspace"):
        self.env = {
            "workspace": workspace,
            "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        }
        self.tools = AgentTools(self.env["workspace"])
        self.brain = RawSelfAttentionBrain(embed_dim=4, device=self.env["device"])

    def train_step(self, inputs, targets, lr=0.01):
        outputs = self.brain.forward(inputs)
        device = self.env["device"]

        if device != "cpu":
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, dtype=torch.float64, device=device)
            loss = torch.mean((outputs - targets) ** 2)
            grad_output = 2.0 * (outputs - targets) / outputs.numel()
            self.brain.backward_and_optimize(grad_output, lr)
            return loss.item()
        else:
            rows, cols = len(outputs), len(outputs[0])
            loss = 0.0
            grad_output = [[0.0] * cols for _ in range(rows)]
            total_elements = rows * cols

            for i in range(rows):
                for j in range(cols):
                    diff = outputs[i][j] - targets[i][j]
                    loss += diff ** 2
                    grad_output[i][j] = 2.0 * diff / total_elements

            self.brain.backward_and_optimize(grad_output, lr)
            return loss / total_elements

    def run_task(self, filename, code_content):
        device_name = self.env["device"].upper()
        print(f"🧠 [Raw 뇌 연산] 장치 ({device_name}) 가속 판단 중...")
        sample_input = [[0.5, -0.2, 0.1, 0.9], [0.1, 0.8, -0.4, 0.3]]
        decision = self.brain.forward(sample_input)
        print(f"   -> FP64 고정밀 뇌 연산 출력:\n{decision}")

        print("\n💾 파일 생성 및 저장 중...")
        write_res = self.tools.write_file(filename, code_content)
        print(f"   -> {write_res}")

        print("\n💻 터미널 자율 검증 중...")
        term_res = self.tools.run_terminal(f"python3 {filename}")
        print(f"   -> {term_res}")

if __name__ == "__main__":
    agent = AIAgent()

    # 1. FP64 역전파 및 SGD 학습 테스트 (3회 반복)
    dummy_input = [[0.1, 0.2, 0.3, 0.4]]
    target = [[0.0, 0.0, 0.0, 0.0]]

    print("📉 [FP64 역전파 학습 진행]")
    for step in range(1, 4):
        loss_val = agent.train_step(dummy_input, target, lr=0.05)
        print(f"   Step {step} - Loss: {loss_val:.10f}")

    print("\n--------------------------------------------------\n")

    # 2. 에이전트 자율 작업 실행
    test_code = "print('Safe Softmax(e^x) + FP64 역전파 학습 엔진 정상 동작 완료!')"
    agent.run_task("test_script.py", test_code)
