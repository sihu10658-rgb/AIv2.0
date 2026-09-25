import os
import math
import random
import subprocess

# ==========================================
# 1. 고정밀(FP64) 뉴런 레이어 (Y = X · W + b + GELU)
# ==========================================
class RawLinearNeuron:
    def __init__(self, in_features, out_features):
        # FP64 가중치(W) 및 편향(b) 초기화
        self.W = [[random.gauss(0, 0.1) for _ in range(out_features)] for _ in range(in_features)]
        self.b = [0.01 for _ in range(out_features)]

    def forward(self, X):
        rows_X, cols_X = len(X), len(X[0])
        cols_W = len(self.W[0])
        
        # Step 1: Y = X · W + b (가중합 + 편향)
        Y = [[0.0] * cols_W for _ in range(rows_X)]
        for i in range(rows_X):
            for k in range(cols_X):
                for j in range(cols_W):
                    Y[i][j] += X[i][k] * self.W[k][j]
                    
        for i in range(rows_X):
            for j in range(cols_W):
                Y[i][j] += self.b[j]

        # Step 2: GELU 비선형 활성화 함수 통과
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = math.sqrt(2.0 / math.pi)
        activated_Y = []
        for row in Y:
            new_row = []
            for x in row:
                inner = sqrt_2_pi * (x + 0.044715 * (x ** 3))
                new_row.append(0.5 * x * (1.0 + math.tanh(inner)))
            activated_Y.append(new_row)

        return activated_Y

# ==========================================
# 2. Raw 트랜스포머 Self-Attention 뇌 연산
# ==========================================
class RawSelfAttentionBrain:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        # Q, K, V 변환용 뉴런 레이어
        self.q_layer = RawLinearNeuron(embed_dim, embed_dim)
        self.k_layer = RawLinearNeuron(embed_dim, embed_dim)
        self.v_layer = RawLinearNeuron(embed_dim, embed_dim)

    def _matmul(self, A, B):
        rows_A, cols_A = len(A), len(A[0])
        cols_B = len(B[0])
        result = [[0.0] * cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for k in range(cols_A):
                for j in range(cols_B):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def _transpose(self, A):
        return [list(i) for i in zip(*A)]

    def _softmax(self, vector):
        max_val = max(vector)
        exps = [math.exp(x - max_val) for x in vector]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]

    def forward(self, X):
        # 1. 뉴런 레이어를 거쳐 Q, K, V 추출 (Y = X·W + b + GELU)
        Q = self.q_layer.forward(X)
        K = self.k_layer.forward(X)
        V = self.v_layer.forward(X)

        # 2. Attention Scores = (Q · K^T) / sqrt(d_k)
        K_T = self._transpose(K)
        scores = self._matmul(Q, K_T)
        scale = math.sqrt(self.embed_dim)
        scaled_scores = [[val / scale for val in row] for row in scores]

        # 3. Softmax 적용
        attn_weights = [self._softmax(row) for row in scaled_scores]

        # 4. Context Output = Weights · V
        output = self._matmul(attn_weights, V)
        return output

# ==========================================
# 3. 파일 및 터미널 검증 도구
# ==========================================
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

# ==========================================
# 4. 통합 에이전트 클래스
# ==========================================
class AIAgent:
    def __init__(self, workspace="./my_workspace"):
        self.tools = AgentTools(workspace)
        self.brain = RawSelfAttentionBrain(embed_dim=4)

    def train_step(self, inputs, targets):
        """FP64 손실 측정 (MSE 수동 계산)"""
        outputs = self.brain.forward(inputs)
        loss = 0.0
        rows, cols = len(outputs), len(outputs[0])
        for i in range(rows):
            for j in range(cols):
                loss += (outputs[i][j] - targets[i][j]) ** 2
        return loss / (rows * cols)

    def run_task(self, filename, code_content):
        print("🧠 [Raw 뇌 연산] Y = X·W + b + GELU + Self-Attention 판단 중...")
        sample_input = [[0.5, -0.2, 0.1, 0.9], [0.1, 0.8, -0.4, 0.3]]
        decision = self.brain.forward(sample_input)
        print(f"   -> FP64 고정밀 뇌 연산 출력: {decision}")

        print("\n💾 파일 생성 및 저장 중...")
        write_res = self.tools.write_file(filename, code_content)
        print(f"   -> {write_res}")

        print("\n💻 터미널 자율 검증 중...")
        term_res = self.tools.run_terminal(f"python3 {filename}")
        print(f"   -> {term_res}")

if __name__ == "__main__":
    agent = AIAgent()

    # 1. FP64 수치 오차 측정
    dummy_input = [[0.1, 0.2, 0.3, 0.4]]
    target = [[0.0, 0.0, 0.0, 0.0]]
    loss_val = agent.train_step(dummy_input, target)
    print(f"📉 [FP64 Raw Loss] 측정 오차: {loss_val:.8f}\n")

    # 2. 에이전트 자율 작업 실행
    test_code = "print('뉴런 + GELU 활성화 함수 + Raw 트랜스포머 실행 성공!')"
    agent.run_task("test_script.py", test_code)
