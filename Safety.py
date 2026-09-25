import shutil
import os

def safe_delete_folder_contents(target_dir):

    allowed_workspace = "/Users/username/projects/"
    abs_target = os.path.abspath(target_dir)
    
    if not abs_target.startswith(allowed_workspace):
        return f"❌ [거부] {target_dir}은 허용된 작업 공간 외부입니다."
        
    shutil.rmtree(abs_target)
    os.makedirs(abs_target)
    return f"✅ {abs_target} 내부가 안전하게 비워졌습니다."
