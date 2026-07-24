FROM ros:humble-ros-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Accept EULA for Microsoft fonts so the Docker build doesn't hang
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3-pip \
    python3-dev \
    # Install MS core fonts (includes Times New Roman) and font config
    ttf-mscorefonts-installer \
    fontconfig \
    # ROS2 related dependencies
    ros-humble-rosbag2 \
    ros-humble-rosbag2-py \
    && rm -rf /var/lib/apt/lists/* \
    # Build the system font cache
    && fc-cache -f -v

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /dynamic-parameter-identification

# Copy project files
COPY . .

# Install Python packages from pyproject.toml
RUN pip install -e .

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create a non-root user for development
RUN useradd -m -s /bin/bash developer && \
    chown -R developer:developer /dynamic-parameter-identification

# Switch to non-root user
USER developer

# Set bash as default shell
SHELL ["/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]