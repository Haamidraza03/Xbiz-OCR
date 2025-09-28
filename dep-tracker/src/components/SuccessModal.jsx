import { Modal, Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';

const SuccessModal = ({ show, onHide, message }) => {
  const navigate = useNavigate();

  return (
    <Modal show={show} onHide={onHide}>
      <Modal.Header closeButton className="bg-success text-white">
        <Modal.Title>
          <i className="fas fa-check-circle me-2"></i>Success!
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p>{message}</p>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
        <Button variant="primary" onClick={() => navigate('/deployment-history')}>
          <i className="fas fa-history me-1"></i>View History
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default SuccessModal;