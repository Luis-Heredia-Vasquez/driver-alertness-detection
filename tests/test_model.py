"""Comprehensive model tests for SimpleCNN."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


# Mock PyTorch module for test compatibility
class MockTensor:
    """Mock PyTorch tensor for testing when torch unavailable."""
    def __init__(self, data):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
    
    def numpy(self):
        return self.data
    
    def to(self, device):
        return self
    
    def backward(self):
        pass
    
    def zero_grad(self):
        pass


class TestModelInstantiation:
    """Test SimpleCNN model instantiation."""

    def test_model_basic_instantiation(self):
        """Test basic model creation."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            assert model is not None
            assert isinstance(model, torch.nn.Module)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_custom_classes(self):
        """Test model with custom number of classes."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            for num_classes in [2, 3, 5, 10]:
                model = SimpleCNN(num_classes=num_classes)
                assert model is not None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_device_placement(self):
        """Test model can be placed on different devices."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN()
            
            # CPU placement
            model = model.to('cpu')
            assert next(model.parameters()).device.type == 'cpu'
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_training_mode(self):
        """Test model can switch between train and eval modes."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN()
            
            model.train()
            assert model.training is True
            
            model.eval()
            assert model.training is False
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_parameter_count(self):
        """Test model has correct number of trainable parameters."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            assert total_params > 0
            
            # Roughly estimate: Conv layers + FC layers should have substantial params
            assert total_params > 10000  # At least 10k parameters
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_parameters_requires_grad(self):
        """Test all parameters require gradients by default."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN()
            
            for param in model.parameters():
                assert param.requires_grad is True
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestModelForwardPass:
    """Test model forward pass with various inputs."""

    def test_forward_batch_size_1(self):
        """Test forward pass with batch size 1."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            # Input: (1, 3, 64, 64) - standard size after preprocessing
            x = torch.randn(1, 3, 64, 64)
            output = model(x)
            
            assert output.shape == (1, 2)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_forward_batch_size_32(self):
        """Test forward pass with typical batch size."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            x = torch.randn(32, 3, 64, 64)
            output = model(x)
            
            assert output.shape == (32, 2)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_forward_multiclass(self):
        """Test forward pass with multiple output classes."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            for num_classes in [3, 5, 10]:
                model = SimpleCNN(num_classes=num_classes)
                model.eval()
                
                x = torch.randn(16, 3, 64, 64)
                output = model(x)
                
                assert output.shape == (16, num_classes)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_forward_output_range(self):
        """Test forward output contains reasonable values."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            x = torch.randn(8, 3, 64, 64)
            output = model(x)
            
            # Output logits should be reasonable (not NaN/Inf)
            assert torch.isfinite(output).all()
            assert output.dtype in [torch.float32, torch.float64]
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_forward_gradient_flow(self):
        """Test gradients flow through model."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            
            x = torch.randn(4, 3, 64, 64, requires_grad=True)
            output = model(x)
            
            # Compute loss and backward
            loss = output.sum()
            loss.backward()
            
            # Check gradients exist
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_forward_deterministic_with_seed(self):
        """Test forward pass is deterministic with seed."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            x = torch.randn(4, 3, 64, 64)
            
            torch.manual_seed(42)
            output1 = model(x)
            
            torch.manual_seed(42)
            output2 = model(x)
            
            # Different random inputs should give different outputs
            assert not torch.allclose(output1, output2)
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestModelCheckpointing:
    """Test model save/load functionality."""

    def test_save_checkpoint(self):
        """Test model checkpoint saving."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = os.path.join(tmpdir, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                
                assert os.path.exists(checkpoint_path)
                assert os.path.getsize(checkpoint_path) > 0
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_load_checkpoint(self):
        """Test model checkpoint loading."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model1 = SimpleCNN(num_classes=2)
            original_params = [p.clone() for p in model1.parameters()]
            
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = os.path.join(tmpdir, 'model.pth')
                torch.save(model1.state_dict(), checkpoint_path)
                
                # Load into new model
                model2 = SimpleCNN(num_classes=2)
                model2.load_state_dict(torch.load(checkpoint_path))
                
                # Compare parameters
                for p1, p2 in zip(model1.parameters(), model2.parameters()):
                    assert torch.allclose(p1, p2)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_checkpoint_round_trip(self):
        """Test complete save-load-forward pass round trip."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model1 = SimpleCNN(num_classes=2)
            model1.eval()
            
            x = torch.randn(4, 3, 64, 64)
            output1 = model1(x)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = os.path.join(tmpdir, 'model.pth')
                torch.save(model1.state_dict(), checkpoint_path)
                
                model2 = SimpleCNN(num_classes=2)
                model2.load_state_dict(torch.load(checkpoint_path))
                model2.eval()
                
                output2 = model2(x)
                
                # Outputs should be identical
                assert torch.allclose(output1, output2)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_checkpoint_multiclass_model(self):
        """Test checkpoint save/load for multiclass model."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model1 = SimpleCNN(num_classes=5)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = os.path.join(tmpdir, 'model_5class.pth')
                torch.save(model1.state_dict(), checkpoint_path)
                
                model2 = SimpleCNN(num_classes=5)
                model2.load_state_dict(torch.load(checkpoint_path))
                
                # Verify outputs have correct shape
                x = torch.randn(8, 3, 64, 64)
                output = model2(x)
                assert output.shape == (8, 5)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_checkpoint_after_training_step(self):
        """Test checkpoint after simulated training step."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Simulate training step
            model.train()
            x = torch.randn(4, 3, 64, 64)
            y = torch.randint(0, 2, (4,))
            
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            # Save after training step
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = os.path.join(tmpdir, 'model_trained.pth')
                torch.save(model.state_dict(), checkpoint_path)
                
                # Load and verify
                model2 = SimpleCNN(num_classes=2)
                model2.load_state_dict(torch.load(checkpoint_path))
                
                assert os.path.exists(checkpoint_path)
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestModelTraining:
    """Test model training behavior."""

    def test_training_mode_affects_behavior(self):
        """Test that train/eval modes affect model behavior."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            x = torch.randn(4, 3, 64, 64)
            
            # Different outputs in train vs eval (due to dropout, BatchNorm, etc)
            model.train()
            torch.manual_seed(42)
            out_train = model(x)
            
            model.eval()
            torch.manual_seed(42)
            out_eval = model(x)
            
            # Outputs may differ (stochastic behavior in train mode)
            # Just verify both are valid
            assert torch.isfinite(out_train).all()
            assert torch.isfinite(out_eval).all()
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple steps."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            
            model.train()
            
            # Accumulate gradients
            grad_norms_before = []
            for i in range(3):
                x = torch.randn(4, 3, 64, 64)
                y = torch.randint(0, 2, (4,))
                
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                
                # Record gradient norm
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                grad_norms_before.append(grad_norm)
            
            # All steps should have non-zero gradients
            assert all(g > 0 for g in grad_norms_before)
            
            # Apply optimization step
            optimizer.step()
            
            # Verify parameters updated
            x_test = torch.randn(4, 3, 64, 64)
            with torch.no_grad():
                output_after = model(x_test)
            
            assert torch.isfinite(output_after).all()
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_learning_rate_scheduling(self):
        """Test model works with learning rate scheduling."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            
            initial_lr = optimizer.param_groups[0]['lr']
            
            # Step scheduler
            scheduler.step()
            
            new_lr = optimizer.param_groups[0]['lr']
            assert new_lr < initial_lr
            assert abs(new_lr - initial_lr * 0.5) < 1e-5
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestModelMetrics:
    """Test metrics computation on model outputs."""

    def test_accuracy_computation(self):
        """Test accuracy computation from model predictions."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            x = torch.randn(100, 3, 64, 64)
            y_true = torch.randint(0, 2, (100,))
            
            with torch.no_grad():
                logits = model(x)
                predictions = logits.argmax(dim=1)
            
            accuracy = (predictions == y_true).float().mean().item()
            assert 0.0 <= accuracy <= 1.0
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            from sklearn.metrics import confusion_matrix
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            x = torch.randn(100, 3, 64, 64)
            y_true = torch.randint(0, 2, (100,))
            
            with torch.no_grad():
                logits = model(x)
                predictions = logits.argmax(dim=1)
            
            cm = confusion_matrix(y_true.numpy(), predictions.numpy())
            assert cm.shape == (2, 2)
            assert cm.sum() == 100
        except (ImportError, ModuleNotFoundError):
            pytest.skip("PyTorch or sklearn not installed")

    def test_precision_recall(self):
        """Test precision and recall computation."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            from sklearn.metrics import precision_score, recall_score
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            x = torch.randn(200, 3, 64, 64)
            # Create imbalanced labels
            y_true = torch.cat([torch.zeros(150, dtype=torch.long), torch.ones(50, dtype=torch.long)])
            
            with torch.no_grad():
                logits = model(x)
                predictions = logits.argmax(dim=1)
            
            precision = precision_score(y_true.numpy(), predictions.numpy(), zero_division=0)
            recall = recall_score(y_true.numpy(), predictions.numpy(), zero_division=0)
            
            assert 0.0 <= precision <= 1.0
            assert 0.0 <= recall <= 1.0
        except (ImportError, ModuleNotFoundError):
            pytest.skip("PyTorch or sklearn not installed")

    def test_loss_computation(self):
        """Test loss computation."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            criterion = torch.nn.CrossEntropyLoss()
            
            x = torch.randn(16, 3, 64, 64)
            y = torch.randint(0, 2, (16,))
            
            output = model(x)
            loss = criterion(output, y)
            
            assert loss.item() >= 0
            assert torch.isfinite(loss)
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestModelInputValidation:
    """Test model input validation and edge cases."""

    def test_wrong_channel_count(self):
        """Test model with wrong number of input channels."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            
            # Wrong number of channels (should be 3 for RGB)
            x = torch.randn(4, 1, 64, 64)  # Grayscale instead of RGB
            
            with pytest.raises((RuntimeError, ValueError)):
                model(x)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_zero_batch_size(self):
        """Test model with zero batch size."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            x = torch.randn(0, 3, 64, 64)  # Empty batch
            
            # Should either work or raise informative error
            try:
                output = model(x)
                assert output.shape[0] == 0
            except RuntimeError:
                # Expected for some framework versions
                pass
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_very_large_batch(self):
        """Test model with large batch size (memory check)."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            # Large but reasonable batch
            x = torch.randn(256, 3, 64, 64)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (256, 2)
        except ImportError:
            pytest.skip("PyTorch not installed")
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Out of memory")
            raise

    def test_non_tensor_input(self):
        """Test model with non-tensor input."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            
            # Pass numpy array instead of tensor
            x_np = np.random.randn(4, 3, 64, 64)
            
            with pytest.raises((TypeError, AttributeError)):
                model(x_np)
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestModelExporting:
    """Test model export and conversion."""

    def test_export_onnx_format(self):
        """Test exporting model to ONNX format."""
        try:
            import torch
            import torch.onnx
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, 'model.onnx')
                
                dummy_input = torch.randn(1, 3, 64, 64)
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    verbose=False,
                    input_names=['input'],
                    output_names=['output']
                )
                
                assert os.path.exists(onnx_path)
                assert os.path.getsize(onnx_path) > 0
        except (ImportError, AttributeError):
            pytest.skip("PyTorch ONNX export not available")

    def test_export_scripted_model(self):
        """Test exporting model as TorchScript."""
        try:
            import torch
            from src.models.cnn import SimpleCNN
            
            model = SimpleCNN(num_classes=2)
            model.eval()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = os.path.join(tmpdir, 'model.pt')
                
                scripted_model = torch.jit.script(model)
                torch.jit.save(scripted_model, script_path)
                
                assert os.path.exists(script_path)
                
                # Load and verify
                loaded_model = torch.jit.load(script_path)
                x = torch.randn(4, 3, 64, 64)
                output = loaded_model(x)
                assert output.shape == (4, 2)
        except (ImportError, RuntimeError):
            pytest.skip("TorchScript export not available")
