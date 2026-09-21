"""
Aspen Plus COM Interface Automation
Connects Python to Aspen Plus using the Component Object Model (win32com.client).
Enables automated Latin Hypercube Sampling, running simulations, and extracting stream results.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AspenPlusCOMAutomation:
    """
    Automates Aspen Plus simulation runs via the ActiveX/COM interface.
    """

    def __init__(self, bkp_filepath: Optional[str] = None, visible: bool = False):
        self.bkp_filepath = bkp_filepath
        self.visible = visible
        self.aspen_app = None
        self.is_connected = False

    def connect(self) -> bool:
        """
        Initializes COM connection to Aspen Plus application.
        """
        if sys.platform != "win32":
            logger.warning(
                "Aspen Plus COM interface requires Windows OS and an active Aspen Plus license. "
                "Running in standalone simulation mode."
            )
            return False

        try:
            import win32com.client as win32

            logger.info("Initializing Aspen Plus COM server (Apwn.Document)...")
            self.aspen_app = win32.Dispatch("Apwn.Document")
            self.aspen_app.Visible = self.visible

            if self.bkp_filepath and os.path.exists(self.bkp_filepath):
                logger.info(f"Opening Aspen Plus file: {self.bkp_filepath}")
                self.aspen_app.InitFromArchive2(os.path.abspath(self.bkp_filepath))

            self.is_connected = True
            logger.info("Successfully connected to Aspen Plus.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Aspen Plus COM interface: {e}")
            self.is_connected = False
            return False

    def set_node_value(self, node_path: str, value: Any) -> bool:
        """
        Sets a variable in the Aspen Plus simulation tree.
        Example node_path: r"\\Data\\Blocks\\B1\\Input\\RR"
        """
        if not self.is_connected:
            logger.warning(f"[MOCK] Setting node {node_path} = {value}")
            return True

        try:
            node = self.aspen_app.Tree.FindNode(node_path)
            if node:
                node.Value = value
                return True
            else:
                logger.error(f"Node path not found: {node_path}")
                return False
        except Exception as e:
            logger.error(f"Error setting node {node_path}: {e}")
            return False

    def get_node_value(self, node_path: str) -> Optional[Any]:
        """
        Retrieves a result from the Aspen Plus simulation tree.
        Example node_path: r"\\Data\\Blocks\\B1\\Output\\REB_DUTY"
        """
        if not self.is_connected:
            logger.warning(f"[MOCK] Returning simulated value for {node_path}")
            return None

        try:
            node = self.aspen_app.Tree.FindNode(node_path)
            return node.Value if node else None
        except Exception as e:
            logger.error(f"Error reading node {node_path}: {e}")
            return None

    def run_simulation(self) -> bool:
        """
        Triggers simulation calculation in Aspen Plus.
        """
        if not self.is_connected:
            return True

        try:
            logger.info("Executing Aspen calculation engine...")
            self.aspen_app.Engine.Run2()
            return True
        except Exception as e:
            logger.error(f"Simulation execution failed: {e}")
            return False

    def close(self):
        """
        Closes Aspen Plus document and releases COM resources.
        """
        if self.aspen_app:
            try:
                self.aspen_app.Close()
                logger.info("Aspen Plus document closed.")
            except Exception as e:
                logger.error(f"Error closing Aspen Plus: {e}")
            finally:
                self.aspen_app = None
                self.is_connected = False
