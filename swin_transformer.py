from __init__ import *
import math
from einops import rearrange


def to_2tuple(arg):
    if isinstance(arg, int):
        return arg, arg
    else:
        if isinstance(arg, (tuple, list)) and len(arg) == 2:
            return arg
        else:
            raise ValueError('arg must be an int or a container of 2 int')



class Image2Embed(nn.Module):

    def __init__(self,
                 img_size: Union[int, tuple],
                 patch_size: Union[int, tuple],
                 embed_dim: int,
                 norm_layer: Optional[nn.Module] = None,
                 positional_bias: Optional[nn.Module] = None):      # class_token ?

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        if img_size[0] % patch_size[0] or img_size[1] % patch_size[1]:
            raise ValueError('image is not divided by patches')

        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        patch_dim = patch_size[0] * patch_size[1] * 3

        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        self.image_to_embedding = nn.Sequential(
            Rearrange('b c (n1 p1) (n2 p2) -> b (n1 n2) p1 p2 c', p1=patch_size[0], p2=patch_size[1]),
            Rearrange('b n p1 p2 c -> b n (p1 p2 c)'),
            nn.Linear(patch_dim, embed_dim)
        )

        self.norm_layer = norm_layer(embed_dim) if norm_layer else None


    def forward(self, image):
        x = self.image_to_embedding(image)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        # if self.positional_bias:          # 아니면 그냥 마지막 Swin_Transformer에서 주기?
        #     x += self.positional_bias

        return x



class Image2Embed_Conv(nn.Module):

    def __init__(self,
                 img_size: Union[int, tuple],
                 patch_size: int,
                 embed_dim: int,
                 norm_layer: Optional[nn.Module] = None,
                 positional_bias: Optional[nn.Module] = None):

        img_size = to_2tuple(img_size)

        if img_size[0] % patch_size or img_size[1] % patch_size:
            raise ValueError('image is not divided by patches')

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        patch_dim = patch_size * patch_size * 3

        super().__init__()

        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        self.image_to_embedding = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c n1 n2 -> b (n1 n2) c')
        )

        self.norm_layer = norm_layer(embed_dim) if norm_layer else None


    def forward(self, image):
        x = self.image_to_embedding(image)

        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x



class Patch2Window(nn.Module):

    def __init__(self,
                 input_size: Union[int, tuple],
                 patch_size: Union[int, tuple],
                 window_size: Union[int, tuple],
                 from_embed: bool = True,
                 to_embed: bool = True):

        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        window_size = to_2tuple(window_size)

        if input_size[0] % patch_size[0] or input_size[1] % patch_size[1]:
            raise ValueError('input is not divided by patches')

        num_patches = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])

        if num_patches[0] % window_size[0] or num_patches[1] % window_size[1]:
            raise ValueError('patches are not divided by windows')

        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.window_size = window_size
        self.from_embed = from_embed
        self.to_embed = to_embed


    def forward(self, patches):
        if self.from_embed:
            patches = rearrange(patches, 'b (n1 n2) c -> b n1 n2 c',
                                n1=self.num_patches[0], n2=self.num_patches[1])
        if self.to_embed:
            windows = rearrange(patches, 'b (n1 w1) (n2 w2) c -> (b n1 n2) (w1 w2) c',
                                w1=self.window_size[0], w2=self.window_size[1])
        else:
            windows = rearrange(patches, 'b (n1 w1) (n2 w2) c -> (b n1 n2) w1 w2 c',
                                w1=self.window_size[0], w2=self.window_size[1])
        return windows



class Window2Patch(nn.Module):

    def __init__(self,
                 input_size: Union[int, tuple],
                 patch_size: Union[int, tuple],
                 window_size: Union[int, tuple],
                 from_embed: bool = True,
                 to_embed: bool = True):

        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        window_size = to_2tuple(window_size)

        if input_size[0] % patch_size[0] or input_size[1] % patch_size[1]:
            raise ValueError('input is not divided by patches')

        num_patches = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])

        if num_patches[0] % window_size[0] or num_patches[1] % window_size[1]:
            raise ValueError('patches are not divided by windows')

        num_windows = (num_patches[0] // window_size[0], num_patches[1] // window_size[1])

        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.window_size = window_size
        self.num_windows = num_windows
        self.from_embed = from_embed
        self.to_embed = to_embed


    def forward(self, windows):

        if self.from_embed and self.to_embed:
            patches = rearrange(windows, '(b n1 n2) (w1 w2) c -> b (n1 w1 n2 w2) c',
                                n1=self.num_windows[0], n2=self.num_windows[1],
                                w1=self.window_size[0], w2=self.window_size[1])

        elif self.from_embed and not self.to_embed:
            patches = rearrange(windows, '(b n1 n2) (w1 w2) c -> b (n1 w1) (n2 w2) c',
                                n1=self.num_windows[0], n2=self.num_windows[1],
                                w1=self.window_size[0], w2=self.window_size[1])

        elif not self.from_embed and self.to_embed:
            patches = rearrange(windows, '(b n1 n2) w1 w2 c -> b (n1 w1 n2 w2) c',
                                n1=self.num_windows[0], n2=self.num_windows[1],
                                w1=self.window_size[0], w2=self.window_size[1])
        else:
            patches = rearrange(windows, '(b n1 n2) w1 w2 c -> b (n1 w1) (n2 w2) c',
                                n1=self.num_windows[0], n2=self.num_windows[1],
                                w1=self.window_size[0], w2=self.window_size[1])
        return patches



class Attention(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 num_head: int,
                 head_dim: int,
                 drop_rate: float = 0,
                 positional_bias: Optional[nn.Module] = None):

        hidden_dim = num_head * head_dim
        self.num_head = num_head
        self.scale = head_dim ** -2

        self.last = not (num_head == 1 and embedding_dim == hidden_dim)

        super(Attention, self).__init__()

        self.q_head = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.k_head = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.v_head = nn.Linear(embedding_dim, hidden_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        if self.last:
            self.project = nn.Sequential(
                nn.Linear(hidden_dim, embedding_dim),
                nn.Dropout(drop_rate))


    def forward(self, x):
        q, k, v = self.q_head(x), self.k_head(x), self.v_head(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_head)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_head)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_head)

        kt = k.transpose(-1, -2)

        attention = self.softmax(torch.matmul(q, kt) * self.scale)

        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if self.last:
            out = self.project(out)
        return out


##



##
def relative_postional_bias(a):
    return

##

class Window_Attention:
    def __init__(self):
        super().__init__()


class MLP:
    def __init__(self):
        super().__init__()


class Shift_Window:
    def __init__(self):
        super().__init__()


class Swin_Transformer_Block:
    def __init__(self):
        super().__init__()


class Patch_Merge:
    def __init__(self):
        super().__init__()


class Swin_Transformer:
    def __init__(self):
        super().__init__()

    def stage(self):
        return


