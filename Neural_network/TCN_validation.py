import torch
import torch.nn as nn
from torchinfo import summary  # For model summary
from tcn import TCN

# Import your TCN implementation with FiLM modifications
# ... [include your TCN implementation with FiLM here] ...

# Test parameters
BATCH_SIZE = 4
TIMESTEPS = 11  # H+1+F = 5+1+5
NUM_INPUTS = 8  # 4 states + 4 references
STATIC_DIM = 6  # Mass, I_xx, I_yy, I_zz, volume, max_thrust

# Create test inputs
state_ref_window = torch.randn(BATCH_SIZE, NUM_INPUTS, TIMESTEPS)
sub_params = torch.randn(BATCH_SIZE, STATIC_DIM)

print("Input shapes:")
print(f"State/Ref Window: {state_ref_window.shape} (batch, features, timesteps)")
print(f"Submarine Params: {sub_params.shape} (batch, static_params)\n")

# Initialize the model
tcn = TCN(
    num_inputs=NUM_INPUTS,
    num_channels=[64, 64, 128],
    static_dim=STATIC_DIM,
    kernel_size=3,
    causal=False,
    use_norm='weight_norm',
    activation='relu',
    use_skip_connections=True,
    film_hidden_dims=[64, 128],
    output_projection=4  # 4 control outputs
)

print("Model Architecture:")
print(tcn)
print("\nModel Parameters:")
print(f"Total Parameters: {sum(p.numel() for p in tcn.parameters()):,}")
print(f"FiLM Generator Parameters: {sum(p.numel() for p in tcn.film_generator.parameters()):,}")

# Test forward pass
with torch.no_grad():
    controls = tcn(x=state_ref_window, theta=sub_params)

print("\nOutput shapes:")
print(f"Controls: {controls.shape} (batch, features, timesteps)")

# Verify output values
print("\nOutput verification:")
print(f"Min thrust_x: {controls[:,0,:].min().item():.4f}")
print(f"Max thrust_x: {controls[:,0,:].max().item():.4f}")
print(f"Output mean: {controls.mean().item():.4f}")
print(f"Output std: {controls.std().item():.4f}")

# Test different submarine parameters
print("\nTesting different submarine configs...")
new_params = torch.randn(BATCH_SIZE, STATIC_DIM)
new_controls = tcn(x=state_ref_window, theta=new_params)

diff = (controls - new_controls).abs().mean()
print(f"Output difference with new params: {diff.item():.4f}")

# Test without parameters (should error or use default)
try:
    print("\nTesting without submarine params...")
    no_param_output = tcn(x=state_ref_window)
except Exception as e:
    print(f"Error as expected: {str(e)}")
    print("Solution: Always provide theta when static_dim is set")

print("\nAll tests completed successfully!")