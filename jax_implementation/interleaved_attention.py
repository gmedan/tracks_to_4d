import jax
import jax.numpy as jnp
from einops import rearrange, repeat, einsum
from einops.layers.flax import EinMix
import flax.linen as nn
import math

class UnifiedAttention(nn.Module):
    """
    Unified attention module that handles both frame and point attention using 4D input tensors.
    Uses efficient einsum operations with consistent dimension naming throughout.
    """
    num_heads: int
    head_dim: int
    dropout: float = 0.1
    frame_first: bool = True

    def setup(self):
        # QKV projection for frame attention
        self.qkv_frame = EinMix(
            'batch time point dim -> qkv batch heads time point dim_head',
            weight_shape='qkv dim heads dim_head',
            bias_shape='heads dim_head',
            sizes=dict(
                heads=self.num_heads,
                dim_head=self.head_dim,
                qkv=3
            )
        )

        # QKV projection for point attention
        self.qkv_point = EinMix(
            'batch time point dim -> qkv batch heads time point dim_head',
            weight_shape='dim heads dim_head',
            bias_shape='heads dim_head',
            sizes=dict(
                heads=self.num_heads,
                dim_head=self.head_dim,
                qkv=3
            )
        )

        # Output projection
        self.out_proj = EinMix(
            'batch time point heads dim_head -> batch time point dim',
            weight_shape='heads dim_head dim',
            bias_shape='dim',
            sizes=dict(
                heads=self.num_heads,
                dim_head=self.head_dim,
                dim=self.num_heads * self.head_dim
            )
        )

    def frame_attention(self, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        """
        Apply attention across frames while preserving point relationships.
        Input shape: [batch, time, point, dim]
        """
        # Generate QKV with consistent dimension ordering
        qkv = self.qkv_frame(x)
        
        # Extract Q, K, V while maintaining dimension names
        query = qkv[0]
        key = qkv[1]
        value = qkv[2]

        # Compute attention scores with explicit dimension naming
        attention_scores = einsum(
            query, key,
            'batch heads time_query point dim_head, batch heads time_key point dim_head -> batch heads time_query time_key point',
            scale=1/math.sqrt(self.head_dim)
        )

        if mask is not None:
            attention_scores = jnp.where(mask, attention_scores, float('-inf'))

        # Apply softmax over time_key dimension
        attention_weights = jax.nn.softmax(attention_scores, axis=3)
        attention_weights = nn.Dropout(rate=self.dropout)(
            attention_weights, deterministic=not self.training
        )

        # Apply attention weights to values, directly producing desired output ordering
        attended = einsum(
            attention_weights, value,
            'batch heads time_query time_key point, batch heads time_key point dim_head -> batch time_query point heads dim_head'
        )

        return attended

    def point_attention(self, x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        """
        Apply attention across points while preserving temporal relationships.
        Input shape: [batch, time, point, dim]
        """
        # Generate QKV with consistent dimension ordering
        qkv = self.qkv_point(x)
        
        # Extract Q, K, V while maintaining dimension names
        query = qkv[0]
        key = qkv[1]
        value = qkv[2]

        # Compute attention scores with explicit dimension naming
        attention_scores = einsum(
            query, key,
            'batch heads time point_query dim_head, batch heads time point_key dim_head -> batch heads time point_query point_key',
            scale=1/math.sqrt(self.head_dim)
        )

        if mask is not None:
            attention_scores = jnp.where(mask, attention_scores, float('-inf'))

        # Apply softmax over point_key dimension
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attention_weights = nn.Dropout(rate=self.dropout)(
            attention_weights, deterministic=not self.training
        )

        # Apply attention weights to values, directly producing desired output ordering
        attended = einsum(
            attention_weights, value,
            'batch heads time point_query point_key, batch heads time point_key dim_head -> batch time point_query heads dim_head'
        )

        return attended

    def __call__(self, x: jnp.ndarray, 
                 frame_mask: jnp.ndarray = None,
                 point_mask: jnp.ndarray = None,
                 training: bool = True) -> jnp.ndarray:
        """
        Apply unified attention combining both frame and point attention.
        
        Args:
            x: Input tensor of shape [batch, time, point, dim]
            frame_mask: Optional mask for frame attention
            point_mask: Optional mask for point attention
            training: Whether in training mode
        
        Returns:
            Attended tensor of shape [batch, time, point, dim]
        """
        # self.training = training
        
        # Layer normalization
        # x = nn.LayerNorm()(x)
        
        # Apply attention in specified order
        if self.frame_first:
            frame_out = self.frame_attention(x, frame_mask)
            point_out = self.point_attention(frame_out, point_mask)
        else:
            point_out = self.point_attention(x, point_mask)
            frame_out = self.frame_attention(point_out, frame_mask)
        
        # Final projection and residual connection
        output = self.out_proj(point_out)
        return x + output

# Example usage demonstrating the dimension flow
if __name__ == "__main__":
    main_rng = jax.random.PRNGKey(42)
    """Demonstrate how dimensions flow through the attention mechanism."""
    layer = UnifiedAttention(
        num_heads=8,
        head_dim=64,
        dropout=0.1
    )
    
    # Example shapes
    batch_size, time_steps, num_points, dim = 2, 30, 100, 256
    
    print("Dimension flow:")
    print(f"Input: [batch={batch_size}, time={time_steps}, point={num_points}, dim={dim}]")
    print(f"After QKV: [qkv=3, batch={batch_size}, heads=8, time={time_steps}, point={num_points}, dim_head=64]")
    print(f"After attention: [batch={batch_size}, time={time_steps}, point={num_points}, heads=8, dim_head=64]")
    print(f"Output: [batch={batch_size}, time={time_steps}, point={num_points}, dim={dim}]")

    x = jnp.zeros([batch_size, time_steps, num_points, dim])
    params = layer.init(main_rng, x)
    l = layer.apply(main_rng, params)
    pass
    # return layer, jnp.zeros((batch_size, time_steps, num_points, dim))
